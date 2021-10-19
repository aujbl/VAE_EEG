from models.resnet1d import resnet1d
import os
import torch
import torch.nn as nn
from configs.config import cfg
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as F
from models import get_classifiers
import torch.nn as nn


class Trainer(object):
    def __init__(self,):
        super().__init__()
        self.model= get_classifiers(cfg.CLASSIFIER.NAME,**cfg.CLASSIFIER.KWARGS).cuda()

        trainable_params = []
        trainable_params += [param for k, param in self.model.named_parameters()]

        self.optimizer = torch.optim.SGD(trainable_params, lr=cfg.TRAIN.LR,
                                         weight_decay=cfg.TRAIN.WEIGHT_DECAY, momentum=cfg.TRAIN.MOMENTUM)
        # self.optimizer = torch.optim.Adam(trainable_params, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.loss_dict = {}

    def train_step(self, x, class_labels, sample_weight=None):
        x = self.model.forward(x)
        self.optimizer.zero_grad()
        loss = self.loss_fn(x, class_labels)
        if sample_weight is not None:
            loss=sample_weight*loss
            loss=loss.mean()
        else:
            loss=loss.mean()
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
        model_name = 'model_e{:02d}.pth'.format(epoch)  # assume the total epoch will not greater than 99
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
