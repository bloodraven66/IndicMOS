"""
The trainer module for the MOS prediction
"""
import os
import torch
import numpy as np
from utils import common
from tqdm import tqdm
from utils.logger import logger
from pathlib import Path

class Trainer():
    """
    Trainer class for the MOS prediction
    """
    def __init__(self, args, model):
        """
        Initialize the trainer
        """
        self.__dict__ = args
        self.mos_model = model
        self.chk_name = common.get_chk_name(args)
        if self.freeze_ssl:
            logger.info("Freezing SSL model")
            for name, param in self.mos_model.named_parameters():
                if name.startswith("ssl_model"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            logger.info("Not freezing SSL model")
        self.mos_model.to(self.device)
        if self.infer:
            return
        self.optimizer = torch.optim.Adam(self.mos_model.parameters(), lr=self.lr)
        self.best_dev_loss = np.inf
        self.args = args
        
        
    def to_device(self, vars):
        """
        Move the variables to the device
        """
        return [v.to(self.device) for v in vars]
        
    def run(self, loader, mode):
        """
        The train / dev / test loop
        """
        self.handle_start_of_epoch(mode)
        if mode == "train":
            self.mos_model.train()
        else:
            self.mos_model.eval()
        
        for batch in tqdm(loader, desc=mode, disable=False if mode=='train' else True):
            audio, score, cer, lengths, langs, mcs, tasks, fname = batch
            audio, score, cer, langs, mcs, tasks = self.to_device((audio, score, cer, langs, mcs, tasks))
            cer_data, lang_data, mc_data, task_data  = None, None, None, None
            if self.use_cer:
                cer_data = cer.float()
            if self.use_lang:
                lang_data = langs
            if self.use_mc:
                mc_data = mcs
            if self.use_task:
                task_data = tasks
                
            output = self.mos_model(audio, lengths, cer_data, lang_data, mc_data, task_data)
            loss = self.mos_model.regression_loss(output, score)
           
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.mse_loss_iter[mode].append(loss.item())
        self.handle_end_of_epoch(mode)
    
    def handle_start_of_epoch(self, mode):
        """
        Initialize the loss for the epoch
        """
        self.mse_loss = {}
        self.mse_loss[mode] = []
        self.mse_loss_iter = {}
        self.mse_loss_iter[mode] = []
        
    def handle_end_of_epoch(self, mode):
        """
        Aggregate the loss for the epoch
        """
        self.mse_loss[mode] = np.mean(self.mse_loss_iter[mode])
        logger.info(f'{mode} - {self.mse_loss[mode]}')
    
    def infer_run(self, loader, save_name):
        """
        The loop for infer mode
        """
        self.mos_model.load_state_dict(torch.load(self.chk_name, map_location=self.device))
        self.mos_model.eval()
        self.mse_loss = []
        results = {}
        
        metrics_name = os.path.join(self.metrics_folder, Path(self.chk_name).stem + '_' + f'test-{save_name}'+'.txt')
        logger.info(f'Saving metrics to {metrics_name}')
        
        for batch in tqdm(loader, desc="infer", disable=False):
            audio, score, cer, lengths, langs, mcs, tasks, fname = batch
            audio, score, cer, langs, mcs, tasks = self.to_device((audio, score, cer, langs, mcs, tasks))
            cer_data, lang_data, mc_data, task_data  = None, None, None, None
            if self.use_cer:
                cer_data = cer.float()
            if self.use_lang:
                lang_data = langs
            if self.use_mc:
                mc_data = mcs
            if self.use_task:
                task_data = tasks
                
            output = self.mos_model(audio, lengths, cer_data, lang_data, mc_data, task_data).squeeze()
            loss = self.mos_model.regression_loss(output, score)
            self.mse_loss.append(loss.item())
            for i in range(len(fname)):
                results[fname[i]] = output[i].squeeze().item()
        self.mse_loss = np.mean(self.mse_loss)
        logger.info(f'infer - {self.mse_loss}')
        
        with open(metrics_name, "w") as f:
            for x in results:
                f.write(f'{x}\t{str(results[x])}\n')
        
def main(args, model, loaders, extra_args=None):
    """
    Main function to train the model
    """
    trainer = Trainer(args, model)
    
    if args.infer:
        trainer.infer_run(loaders[2], save_name=args.data.test_dataset[0])
        trainer.infer_run(loaders[3], save_name=args.data.test_dataset[1])
        exit()
    
    for ep in range(args.num_epochs):
        trainer.run(loaders[0], mode='train')
        trainer.run(loaders[1], mode='dev')
        
        if trainer.mse_loss['dev'] < trainer.best_dev_loss:
            
            if ep > args.save_patience:
                trainer.best_dev_loss = trainer.mse_loss['dev']
                torch.save(trainer.mos_model.state_dict(), trainer.chk_name)
                logger.info(f'Saved checkpoint to {trainer.chk_name} at epoch {ep}')
            else:
                logger.info(f'Not saving checkpoint as patience is {ep}')
            
        trainer.run(loaders[2], mode='test')
        trainer.run(loaders[3], mode='test')
        