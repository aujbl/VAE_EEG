import argparse
import datetime
import json
import logging
import os
import random
import sys
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from datasets.dataset import TrainDataset
# from datasets.graph_dataset import GraphTrainDataset as TrainDataset
from trainer import Trainer
from utils.log_helper import init_log, log_grads, track
from utils.misc import AverageMeter, compute_eta_time, mkdir, set_seed
from eval import evaluate
from configs.config import cfg
from transferprob import transferprob

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str, help='which config file to use')
args = parser.parse_args()




def train(trainer, dataloader, val_dataloader, file_writer):
    best_f1_micro = 0
    best_epoch = 0
    num_batch_per_epoch = len(dataloader)
    for epoch in range(cfg.TRAIN.START_EPOCH+1, cfg.TRAIN.END_EPOCH+1):
        trainer.model.train()
        train_loss = AverageMeter()
        batch_time = AverageMeter()
        iter_begin = time.time()
        epoch_begin = time.time()
        train_time = 0
        val_time = 0
        for iter, data in enumerate(track(dataloader)):
            seqs = data['seq'].cuda()
            labels = data['label'].cuda()

            # seq_np=seqs.detach().cpu().numpy()
            # trans_weight=transferprob(seq_np)
            # trans_weight=torch.from_numpy(trans_weight).cuda()

            trainer.train_step(seqs, labels)

            train_loss.update(trainer.loss_dict['total_loss'])
            batch_time.update(time.time()-iter_begin)
            iter_begin = time.time()

            if iter % cfg.TRAIN.LOG_INTERVAL == 0:
                step = (epoch-1) * num_batch_per_epoch + iter
                for k, v in trainer.loss_dict.items():
                    file_writer.add_scalar(k, v, global_step=step)

            if iter % cfg.TRAIN.PRINT_SPEED_INTERVAL == 0:
                step = (epoch-1) * num_batch_per_epoch + iter

                total_step = cfg.TRAIN.END_EPOCH * num_batch_per_epoch
                eta_day, eta_hour, eta_min = compute_eta_time(step,
                                                              batch_time.avg,
                                                              total_step)
                info = f'Progress: {step:d} / {total_step} [{step/total_step:.2%}], remaining time: {eta_day:d}:{eta_hour:02d}:{eta_min:02d} (D:H:M)'
                for k, v in trainer.loss_dict.items():
                    info += f', {k}: {v:.6f}'
                logging.info(info)

        logging.info(
            f'Experiment: {cfg.EXPERIMENT}, epoch: {epoch}, avg train loss: {train_loss.avg:.6f}, lr: {trainer.get_last_lr()}')

        trainer.update_learning_rate()

        train_time = time.time()-epoch_begin

        if epoch >= cfg.TRAIN.VAL_EPOCH:
            val_begin = time.time()
            if cfg.TRAIN.EVAL_IN_TRAINSET:
                results = evaluate(trainer, dataloader)
                train_top1_accuracy = results['top1']
                train_f1_macro = results['f1_macro']
                train_f1_micro = results['f1_micro']
                logging.info(
                    f'Experiment: {cfg.EXPERIMENT}, epoch: {epoch}, '
                    f'train top1 accuracy: {train_top1_accuracy:.6f}, train f1 macro: {train_f1_macro:.6f}, train f1 micro: {train_f1_micro:.6f}')
            results = evaluate(trainer, val_dataloader)
            top1_accuracy = results['top1']
            f1_macro = results['f1_macro']
            f1_micro = results['f1_micro']

            logging.info(
                f'Experiment: {cfg.EXPERIMENT}, epoch: {epoch}, train_f1_macro: {train_f1_macro: .6f}, train_f1_micro: {train_f1_micro: .6f}, top1 accuracy: {top1_accuracy:.6f}, f1 macro: {f1_macro:.6f}, f1 micro: {f1_micro:.6f}')

            file_writer.add_scalar('train_f1_macro', train_f1_macro, global_step=epoch)
            file_writer.add_scalar('train_f1_micro', train_f1_micro, global_step=epoch)
            file_writer.add_scalar('top1_accuracy', top1_accuracy, global_step=epoch)
            file_writer.add_scalar('f1_macro', f1_macro, global_step=epoch)
            file_writer.add_scalar('f1_micro', f1_micro, global_step=epoch)

            if f1_micro > best_f1_micro:
                best_f1_micro = f1_micro
                best_epoch = epoch
            checkpoint_dir = os.path.join(cfg.TRAIN.LOG_DIR, cfg.EXPERIMENT, 'checkpoints')
            trainer.save_model(checkpoint_dir, epoch)
            val_time = time.time()-val_begin
        logging.info(
            f'Experiment: {cfg.EXPERIMENT}, epoch: {epoch}, train time: {datetime.timedelta(seconds=int(train_time))}, val time: {datetime.timedelta(seconds=int(val_time))}')

    logging.info(f'Experiment: {cfg.EXPERIMENT}, best epoch: {best_epoch}, best f1 micro: {best_f1_micro}')


def main():
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in cfg.GPU_IDS])

    set_seed(cfg.RANDOM_SEED)

    log_dir = os.path.join(cfg.TRAIN.LOG_DIR, cfg.EXPERIMENT, 'logs')
    mkdir(log_dir)
    checkpoint_dir = os.path.join(cfg.TRAIN.LOG_DIR, cfg.EXPERIMENT, 'checkpoints')
    mkdir(checkpoint_dir)

    init_log(log_dir=log_dir)
    logging.info(json.dumps(cfg, indent=4))
    file_writer = SummaryWriter(log_dir)

    train_dataset = TrainDataset(data_root=cfg.DATASET.DATA_ROOT, data_type='train')

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True, num_workers=1)

    val_dataset = TrainDataset(data_root=cfg.DATASET.DATA_ROOT, data_type='val')
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False, num_workers=1)

    trainer = Trainer()

    if cfg.TRAIN.RESUME:
        cfg.TRAIN.START_EPOCH = trainer.resume_model(cfg.TRAIN.RESUME_PATH)
    train(trainer, train_dataloader, val_dataloader, file_writer)


if __name__ == '__main__':
    main()
