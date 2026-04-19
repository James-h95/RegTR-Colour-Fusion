import logging
import os
import sys
import time
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cvhelpers.misc import pretty_time_delta
from cvhelpers.torch_helpers import all_to_device, all_isfinite,\
    CheckPointManager, TorchDebugger

from utils.misc import StatsMeter
from models.generic_model import GenericModel
from utils.misc import metrics_to_string


class Trainer:
    """Generic trainer class. This is inspired from the trainer class in
    Pytorch-lightning. But I did not like the way it handles checkpoints and
    summary logging so I reimplemented a simple version of it"""

    def __init__(self, opt, niter, grad_clip=0.0, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.opt = opt
        self.train_writer = SummaryWriter(os.path.join(self.opt.log_path, 'train'),
                                          flush_secs=10)
        self.val_writer = SummaryWriter(os.path.join(self.opt.log_path, 'val'),
                                        flush_secs=10)
        self.saver = CheckPointManager(os.path.join(self.opt.log_path, 'ckpt', 'model'),
                                       max_to_keep=6, keep_checkpoint_every_n_hours=3.0)
        self.niter = niter
        self.grad_clip = grad_clip
        self.log_path = self.opt.log_path

    def fit(self, model: GenericModel, train_loader, val_loader=None):

        # Setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            self.logger.warning('Using CPU for training. This can be slow...')
        model.to(device)
        model.configure_optimizers()
        model.set_trainer(self)

        # Initialize checkpoint manager and resume from checkpoint if necessary
        if self.opt.resume is not None:
            first_step = global_step = \
                self.saver.load(self.opt.resume, model,
                                optimizer=model.optimizer, scheduler=model.scheduler)
        else:
            first_step = global_step = 0
        # Configure anomaly detection
        torch.autograd.set_detect_anomaly(self.opt.debug)

        done = False
        epoch = 0
        loss_smooth = None
        stats_meter = StatsMeter()
        total_iter = self.niter if self.niter > 0 else len(train_loader) * -self.niter
        train_output, losses = {}, {}

        if self.opt.validate_every < 0:
            # validation interval given in epochs, so convert to steps
            self.opt.validate_every = -self.opt.validate_every * len(train_loader)
            self.logger.info('Validation interval set to {} steps'.format(self.opt.validate_every))

        # Run validation and exit if validate_every = 0
        if self.opt.validate_every == 0:
            self._run_validation(model, val_loader, step=global_step, save_ckpt=False)
            return

        # Validation dry run for sanity checks
        if self.opt.nb_sanity_val_steps > 0:
            self._run_validation(model, val_loader, step=global_step,
                                 limit_steps=self.opt.nb_sanity_val_steps)

        # Main training loop
        while not done:  # Loop over epochs

            self.logger.info('Starting epoch {} (steps {} - {})'.format(
                epoch, global_step, global_step + len(train_loader)))
            tbar = tqdm(total=len(train_loader), ncols=80, smoothing=0)

            # Train
            model.train()
            torch.set_grad_enabled(True)
            model.train_epoch_start()
            t_epoch_start = time.perf_counter()

            for batch_idx, batch in enumerate(train_loader):

                global_step += 1

                # train step
                try:
                    batch = all_to_device(batch, device)
                    train_output, losses = model.training_step(batch, global_step)

                    # Skip backward/step entirely on non-finite loss so NaN
                    # gradients never reach the optimizer. A rare numerically
                    # unstable batch should not be allowed to poison the
                    # model weights via clip_grad_norm_ + step().
                    loss_is_finite = (
                        'total' in losses
                        and losses['total'].requires_grad
                        and all_isfinite(losses['total'])
                    )

                    if model.optimizer_handled_by_trainer:
                        if model.optimizer is not None:
                            model.optimizer.zero_grad()

                        if loss_is_finite:
                            if self.opt.debug:
                                with TorchDebugger():
                                    losses['total'].backward()
                            else:
                                losses['total'].backward()

                            if self.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)

                            if model.optimizer is not None:
                                model.optimizer.step()
                                model.scheduler.step()

                    # Increment counters
                    for k in losses:
                        stats_meter[k].update(losses[k])

                    if loss_smooth is None and loss_is_finite:
                        loss_smooth = losses['total'].item()
                    elif 'total' in losses and not all_isfinite(losses['total']):
                        self._bad_loss_count = getattr(self, '_bad_loss_count', 0) + 1
                        if self._bad_loss_count <= 3 or self._bad_loss_count % 200 == 0:
                            self.logger.warning(
                                'Total loss is not finite, skipping backward '
                                '(count=%d). Instance %s src=%s tgt=%s',
                                self._bad_loss_count,
                                batch.get('idx'), batch.get('src_path'),
                                batch.get('tgt_path'))
                    elif loss_is_finite and loss_smooth is not None:
                        loss_smooth = 0.99 * loss_smooth + 0.01 * losses['total'].item()
                    if loss_smooth is not None:
                        tbar.set_description('Loss:{:.3g}'.format(loss_smooth))

                except Exception as inst:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    # Walk to the deepest frame so the log points at the
                    # actual raise site (e.g. a bare `assert` in a nested
                    # block), not at this try/except.
                    deepest_tb = exc_tb
                    while deepest_tb.tb_next is not None:
                        deepest_tb = deepest_tb.tb_next
                    fname = os.path.split(deepest_tb.tb_frame.f_code.co_filename)[1]
                    # Rate-limit verbose tracebacks: keep them for the first
                    # few occurrences of each (file, line), then just emit a
                    # one-line summary so the log doesn't drown in stack
                    # traces for a known, recoverable skip.
                    key = (fname, deepest_tb.tb_lineno, exc_type.__name__)
                    if not hasattr(self, '_exc_counts'):
                        self._exc_counts = {}
                    self._exc_counts[key] = self._exc_counts.get(key, 0) + 1
                    n = self._exc_counts[key]
                    header = f'{exc_type.__name__} at {fname}:{deepest_tb.tb_lineno} - {inst!r} [#{n}]'
                    if n <= 3:
                        self.logger.error(f'{header}\n{traceback.format_exc()}')
                    elif n % 200 == 0:
                        self.logger.error(f'{header} (suppressing traceback; {n} total so far)')

                tbar.update(1)
                # torch.cuda.empty_cache()

                if global_step == first_step + 1 or global_step % self.opt.summary_every == 0:
                    model.train_summary_fn(writer=self.train_writer, step=global_step,
                                           data_batch=batch, train_output=train_output, train_losses=losses)

                if global_step % self.opt.validate_every == 0:
                    tbar.close()  # we turn off the training progress bar since certain
                                  # environments (e.g. Pycharm) do not handle stacking well

                    # Run validation, and save checkpoint.
                    self._run_validation(model, val_loader, step=global_step)
                    tbar = tqdm(total=len(train_loader), ncols=80, initial=batch_idx+1,
                                desc=tbar.desc[:-2])

                if global_step - first_step >= total_iter:
                    done = True
                    break

            model.train_epoch_end()
            tbar.close()

            losses_dict = {k: stats_meter[k].avg for k in stats_meter}
            log_str = 'Epoch {} complete in {}. Average train losses: '.format(
                epoch, pretty_time_delta(time.perf_counter() - t_epoch_start))
            log_str += metrics_to_string(losses_dict) + '\n'
            self.logger.info(log_str)
            stats_meter.clear()

            epoch += 1

        self.logger.info('Ending training. Number of training steps = {}'.format(global_step))

    def test(self, model: GenericModel, test_loader):
        # Setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            self.logger.warning('Using CPU for training. This can be slow...')
        model.to(device)
        model.set_trainer(self)

        # Initialize checkpoint manager and resume from checkpoint if necessary
        if self.opt.resume is not None and len(self.opt.resume) > 0:
            self.saver.load(self.opt.resume, model)
        else:
            self.logger.warning('No checkpoint given. Will perform inference '
                                'using random weights')

        # Run validation and exit if validate_every = 0
        model.eval()
        test_out_all = []
        with torch.no_grad():

            model.test_epoch_start()

            tbar_test = tqdm(total=len(test_loader), ncols=80, leave=False)
            for test_batch_idx, test_batch in enumerate(test_loader):
                test_batch = all_to_device(test_batch, model.device)
                test_out = model.test_step(test_batch, test_batch_idx)
                test_out_all.append(test_out)
                tbar_test.update(1)
            tbar_test.close()

            model.test_epoch_end(test_out_all)

        model.train()

    def _run_validation(self, model: GenericModel, val_loader, step, limit_steps=-1,
                        save_ckpt=True):
        """Run validation on data from the validation data loader

        Args:
            model: Model
            val_loader: Validation data loader. If None, will skip validation
            limit_steps: If positive, will only run this number of validation
              steps. Useful as a sanity check on the validation code.
            save_ckpt: Whether to save checkpoint at the end

        Returns:
            val_score: Used for selecting the best checkpoint
        """
        if val_loader is None:
            return 0.0

        if limit_steps > 0:
            num_steps = limit_steps
            self.logger.info(f'Performing validation dry run with {num_steps} steps')
        else:
            num_steps = len(val_loader)
            self.logger.info(f'Running validation (step {step})...')

        model.eval()
        val_out_all = []
        with torch.no_grad():

            model.validation_epoch_start()

            tbar_val = tqdm(total=num_steps, ncols=80, leave=False)
            for val_batch_idx, val_batch in enumerate(val_loader):
                if val_batch_idx >= num_steps:
                    break

                val_batch = all_to_device(val_batch, model.device)
                val_out = model.validation_step(val_batch, val_batch_idx)
                val_out_all.append(val_out)
                tbar_val.update(1)
            tbar_val.close()

            val_score, val_outputs = model.validation_epoch_end(val_out_all)
            model.validation_summary_fn(self.val_writer, step, val_outputs)

            log_str = ['Validation ended:']
            if 'losses' in val_outputs:
                log_str.append(metrics_to_string(val_outputs['losses'], '[Losses]'))
            if 'metrics' in val_outputs:
                log_str.append(metrics_to_string(val_outputs['metrics'], '[Metrics]'))
            log_str = '\n'.join(log_str)
            self.logger.info(log_str)

        if save_ckpt:
            self.saver.save(model, step, val_score,
                            optimizer=model.optimizer, scheduler=model.scheduler)

        model.train()

