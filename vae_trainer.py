from my_vae import MyVAE
import os
import torch
import torch.nn as nn
from configs.config import cfg
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

lr = 0.001
batch_size = 64



class VAETrainer(object):
    def __init__(self):
        super().__init__()
        self.model = MyVAE(in_channels=1, latent_dim=128, channels=[1, 32, 32, 64, 64, 128, 128])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.8)

        self.loss_dict = {}

    def train_step(self, x, labels, sample_weight=None):
        args = self.model(x)
        self.optimizer.zero_grad()
        loss = self.model.loss_function(*args, labels)
        if sample_weight is not None:
            loss *= sample_weight
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.loss_dict['total_loss'] = loss

    def update_learning_rate(self):
        self.lr_scheduler.step()

    def get_last_lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def save_model(self, save_dir, epoch):
        model_state = self.model.state_dict()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_name = 'vae_epoch_' + str(epoch)
        ckpt_path = os.path.join(save_dir, model_name)
        ckpt_state = {
            'epoch': epoch,
            'model': model_state,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(ckpt_state, ckpt_path)

    def load_model(self, model_path):
        saved_state_dict = torch.load(model_path)
        if 'model' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model'], strict=False)
        else:
            self.model.load_state_dict(saved_state_dict, strict=False)

    def resume_model(self, model_path):
        saved_state_dict = torch.load(model_path)
        self.model.load_state_dict(saved_state_dict['model'])
        self.optimizer.load_state_dict(saved_state_dict['optimizer'])
        resume_epoch = saved_state_dict['epoch']
        self.lr_scheduler.step(resume_epoch+1)
        return resume_epoch