import importlib

from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import layers.normalization
import settings
from data_provider.data_factory import get_dataset
from experiments.exp_basic import Exp_Basic
from util.tools import EarlyStopping, adjust_learning_rate, test_params_flop, load_model_compile
from util.metrics import metric, update_metrics, calculate_metrics

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Simple(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self, model=None):
        if model is None:
            if self.args.model.endswith('_Ensemble'):
                model = importlib.import_module(f'models.{self.args.model[:-len("_Ensemble")]}').Model_Ensemble(
                    self.args).float()
            else:
                model = importlib.import_module(f'models.{self.args.model}').Model(self.args).float()
        if self.args.normalization: 
            model = layers.normalization.ForecastModel(model, num_features=self.args.enc_in, seq_len=self.args.seq_len,
                                                       process_method=self.args.normalization)
        if hasattr(self.args, 'load_path'):
            if not self.args.freeze: 
                self.model_optim = self._select_optimizer(
                    model=model.to(self.device))  # Otherwise, no need to reload its optimizer
            self.args.logger.info('Load checkpoints from {}'.format(self.args.load_path))
            model = self.load_checkpoint(self.args.load_path, model)
            if self.model_optim is not None:
                self.args.logger.info('Learning rate of model_optim is {}'.format(self.model_optim.param_groups[0]['lr']))
            if self.args.freeze:
                model.requires_grad_(False)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        elif self.args.local_rank != -1: 
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                        find_unused_parameters=self.args.find_unused_parameters)
            
        if torch.__version__ >= '2' and self.args.compile:
            self.args.logger.info('Compile the model by Pytorch 2.0')
            model = torch.compile(model)
        return model

    def _process_batch(self, batch): 
        batch = super()._process_batch(batch)
        batch_x, batch_y = batch[:2] 
        if self.args.model in settings.need_x_y_mark:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]

            # decoder input
            dec_inp = torch.zeros_like(batch_x[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1)

            inp = [batch_x, batch_x_mark, dec_inp, batch_y_mark] + batch[4:]
        elif self.args.model in settings.need_x_mark:
            batch = batch[:3] + batch[4:]
            inp = [batch_x] + batch[2:] 
        else:
            batch = batch[:2] + batch[4:]
            inp = [batch_x] + batch[2:] 
        return inp

    def vali(self, vali_data, vali_loader, criterion):
        self.phase = 'val'
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                outputs = self.forward(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)
                loss = criterion(outputs, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        if self.args.local_rank != -1:
            total_loss = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()
        return total_loss

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        if self.args.normalization is not None and self.args.normalization.lower() == 'dishts' and self.args.alpha > 0:
            loss += F.mse_loss(self.model.processor.phih,
                               batch[self.label_position].mean(1, keepdims=True)) * self.args.alpha
        return loss

    def train(self, setting, train_data=None, train_loader=None, vali_data=None, vali_loader=None):
        if train_data is None:
            train_data, train_loader = self._get_data(flag='train')
        if vali_data is None:
            vali_data, vali_loader = self._get_data(flag='val')

        if self.args.checkpoints:
            path = os.path.join(self.args.checkpoints, setting)
        else:
            path = None

        time_now = time.time()

        train_steps = len(train_loader) 
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if self.args.lradj == 'TST': 
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        else:
            scheduler = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            if self.args.local_rank != -1:
                train_loader.sampler.set_epoch(epoch)
                vali_loader.sampler.set_epoch(epoch)
            
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                self.phase = 'train'
                iter_count += 1
                loss, _ = self._update(batch, criterion, model_optim, scaler)
                train_loss.append(loss.item())

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            train_loss = np.average(train_loss)
            self.args.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            if not self.args.train_only:
                if epoch >= self.args.begin_valid_epoch:
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    # test_loss = self.vali(test_data, test_loader, criterion)
                    self.args.logger.info("Vali Loss: {:.7f}".format(vali_loss))
                    early_stopping(vali_loss, self, path)
                else:
                    self.args.logger.info()
            else:
                early_stopping(train_loss, self, path)
            self.args.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            if early_stopping.early_stop:
                self.args.logger.info("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                self.args.logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if self.args.train_epochs > 0:
            self.args.logger.info('Best Valid MSE: {}'.format(-early_stopping.best_score))
            self.load_state_dict(early_stopping.best_checkpoint,
                                 strict=not (hasattr(self.args, 'freeze') and self.args.freeze))
            if path and self.args.local_rank <= 0:
                if not os.path.exists(path):
                    os.makedirs(path)
                self.args.logger.info('Save checkpoint to {}'.format(path))
                torch.save(self.state_dict(local_rank=self.args.local_rank), path + '/' + 'checkpoint.pth')
        return self.model, train_data, train_loader, vali_data, vali_loader

    def test(self, setting, test_data=None, test_loader=None, test=0, target_variate=None):
        self.phase = 'test'
        if test_data is None:
            test_data, test_loader = self._get_data(flag='test')

        if test:
            path = os.path.join("checkpoints", setting, 'checkpoint.pth')
            self.args.logger.info('Loading {}'.format(path))
            self.load_checkpoint(path)

        self.model.eval()
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                outputs = self.forward(batch)
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)
                update_metrics(outputs, true, statistics, target_variate)

        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        self.args.logger.info('mse:{}, mae:{}'.format(mse, mae))
        return mse, mae, test_data, test_loader

    def predict(self, setting, load=False):
        if load:
            path = os.path.join("checkpoints", setting, 'checkpoint.pth')
            print('Loading', path)
            self.load_checkpoint(path) 

        preds = []
        ground_truths = []
        historys = []
        self.model.eval()                                           
        for flag in ['test']: 
            data_set = get_dataset(self.args, flag, self.device,
                                   wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)
            dataloader = DataLoader(
                data_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False,
                pin_memory=False, )
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    outputs = self.forward(batch) # outputs:[B,L,C]
                    history = batch[0]
                    history = history.cpu().numpy()
                    ground_truth = batch[1]
                    ground_truth = ground_truth.cpu().numpy()
                    pred = outputs.detach().cpu().numpy()  # .squeeze()
                    if self.args.save_all:
                        preds.append(pred)
                    elif i % self.args.save_step == 0:
                        preds.append(pred[0][np.newaxis, :, :]) # add a new axis to make it 3D
                        ground_truths.append(ground_truth[0][np.newaxis, :, :])
                        historys.append(history[0][np.newaxis, :, :])
        preds = np.vstack(preds)
        ground_truths = np.vstack(ground_truths)
        historys = np.vstack(historys)
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'final_prediction.npy', preds)
        np.save(folder_path + 'ground_truth.npy', ground_truths)
        np.save(folder_path + 'history.npy', historys)
        return

    def analysis(self):
        pred_data, pred_loader = self._get_data(flag='test')

        times = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                batch = [d.to(self.device) for d in batch]
                start_time = time.time()
                outputs = self.forward(batch)
                if i > 10:
                    times.append(time.time() - start_time)
                if i == 20:
                    break
        print(times)
        print('Forward Time', sum(times) / len(times))
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))
