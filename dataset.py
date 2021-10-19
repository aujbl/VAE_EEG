import os
import json
import random
import numpy as np
import torch.utils.data as data
from sklearn.preprocessing import minmax_scale
from scipy import signal


class TrainDataset(data.Dataset):
    """

    """

    def __init__(self, data_root, data_type='train'):
        self.data_root = data_root
        self.data_type = data_type
        if data_type == 'train':
            with open(os.path.join(data_root, 'train', 'train_f5.json'), 'r') as f:
                annos = json.load(f)
        elif data_type == 'val':
            with open(os.path.join(data_root, 'train', 'val_f5.json'), 'r') as f:
                annos = json.load(f)
        else:
            raise ValueError(f'The data_type must be in train or val , but get {data_type}')

        self.sample_path = annos['samples']
        self.labels = annos['labels']

    def label2num(self, label_str):
        # different from the official rules, need add 1 if submit
        label_dict = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3}
        label = label_str.split('_')[0]
        return label_dict[label]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        seq_path = os.path.join(self.data_root, self.sample_path[index])
        seq = np.load(seq_path)
        # downsampling
        down_idx = np.arange(0, 1000, 4)
        seq = seq[:, down_idx]
        # b, a = signal.butter(8, 0.375, btype='lowpass')
        # seq = signal.filtfilt(b, a, seq, axis=-1)
        # seq = seq.astype(np.float32)
        # normalize

        # seq = minmax_scale(seq, axis=1)

        # seq = seq[:, :224]
        # horizon flip
        # r = random.random()
        # if r > 0.5:
        #     seq[:, ::-1]
        # vertical flip
        # r = random.random()
        # if r > 0.5:
        #     seq = 1-seq

        label = self.labels[index]
        label = self.label2num(label)
        return {
            'seq': seq,
            'label': label
        }


class TestDataset(data.Dataset):
    """

    """

    def __init__(self, data_root):
        self.data_root = data_root
        self.seq_names = sorted(os.listdir(data_root))

    def __len__(self) -> int:
        return len(self.seq_names)

    def __getitem__(self, index: int):
        seq_path = os.path.join(self.data_root, self.seq_names[index])
        seq = np.load(seq_path)  # 68*1000
        # downsampling
        down_idx = np.arange(0, 1000, 4)
        seq = seq[:, down_idx]

        # b, a = signal.butter(8, 0.375, btype='lowpass')
        # seq = signal.filtfilt(b, a, seq, axis=-1)
        # seq = seq.astype(np.float32)
        # seq = seq[:, :224]
        # normalize
        seq = minmax_scale(seq, axis=1)
        return {
            'seq': seq,
        }
