from my_vae import MyVAE
import os
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lr = 0.0025
latent_dim = 512


class VAETrainer(object):
    def __init__(self):
        super().__init__()
        self.model = MyVAE(in_channels=1, latent_dim=latent_dim, channels=[1, 32, 32, 64, 64, 128, 128])
        self.model.to(device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.8)

        self.loss_dict = {}

    def train_step(self, x, labels, sample_weight=None):
        args = self.model(x)
        self.optimizer.zero_grad()
        loss_dict = self.model.loss_function(*args, label=labels)
        loss = loss_dict['total_loss']
        if sample_weight is not None:
            loss *= sample_weight
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.loss_dict = loss_dict

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