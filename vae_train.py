import os
import sys
import time
import torch
import logging
import datetime
import numpy as np
from vae_trainer import VAETrainer
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append('..')
# from eval import evaluate
from datasets.dataset import TrainDataset
from utils.eval import accuracy, f1_score
from utils.log_helper import init_log, log_grads, track
from utils.misc import AverageMeter, compute_eta_time, mkdir, set_seed

TRAIN_START_EPOCH = 0
TRAIN_END_EPOCH = 20
TRAIN_LOG_INTERVAL = 50
TRAIN_PRINT_SPEED_INTERVAL = 50
TRAIN_VAL_EPOCH = 0
EXPERIMENT = "vae_baseline"
TRAIN_EVAL_IN_TRAINSET = True
TRAIN_LOG_DIR = './logs'
GPU_IDS = [0]
RANDOM_SEED = 520
DATASET_DATA_ROOT = '../../nor_split_datasets'
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
TRAIN_RESUME = False
TRAIN_RESUME_PATH = ''


def train(trainer, dataloader, val_dataloader, file_writer):
    best_f1_micro = 0
    best_epoch = 0
    num_batch_per_epoch = len(dataloader)
    for epoch in range(TRAIN_START_EPOCH+1, TRAIN_END_EPOCH+1):
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

            trainer.train_step(seqs, labels)

            train_loss.update(trainer.loss_dict['total_loss'])
            batch_time.update(time.time()-iter_begin)
            iter_begin = time.time()

            if iter % TRAIN_LOG_INTERVAL == 0:
                step = (epoch-1) * num_batch_per_epoch + iter
                for k, v in trainer.loss_dict.items():
                    file_writer.add_scalar(k, v, global_step=step)

            if iter % TRAIN_PRINT_SPEED_INTERVAL == 0:
                step = (epoch-1) * num_batch_per_epoch + iter

                total_step = TRAIN_END_EPOCH * num_batch_per_epoch
                eta_day, eta_hour, eta_min = compute_eta_time(step,
                                                              batch_time.avg,
                                                              total_step)
                info = f'Progress: {step:d} / {total_step} [{step/total_step:.2%}], ' \
                       f'remaining time: {eta_day:d}:{eta_hour:02d}:{eta_min:02d} (D:H:M)'
                for k, v in trainer.loss_dict.items():
                    info += f', {k}: {v:.6f}'
                logging.info(info)

        logging.info(
            f'Experiment: {EXPERIMENT}, epoch: {epoch},'
            f' avg train loss: {train_loss.avg:.6f}, lr: {trainer.get_last_lr()}'
        )

        trainer.update_learning_rate()
        train_time = time.time()-epoch_begin

        if epoch >= TRAIN_VAL_EPOCH:
            val_begin = time.time()
            if TRAIN_EVAL_IN_TRAINSET:
                results = evaluate(trainer, dataloader)
                train_top1_accuracy = results['top1']
                train_f1_macro = results['f1_macro']
                train_f1_micro = results['f1_micro']
                logging.info(
                    f'Experiment: {EXPERIMENT}, epoch: {epoch}, '
                    f'train top1 accuracy: {train_top1_accuracy:.6f},'
                    f' train f1 macro: {train_f1_macro:.6f}, train f1 micro: {train_f1_micro:.6f}')
            results = evaluate(trainer, val_dataloader)
            top1_accuracy = results['top1']
            f1_macro = results['f1_macro']
            f1_micro = results['f1_micro']

            logging.info(
                f'Experiment: {EXPERIMENT}, epoch: {epoch},'
                f' train_f1_macro: {train_f1_macro: .6f},'
                f' train_f1_micro: {train_f1_micro: .6f}, '
                f'top1 accuracy: {top1_accuracy:.6f}, '
                f'f1 macro: {f1_macro:.6f}, '
                f'f1 micro: {f1_micro:.6f}'
            )

            file_writer.add_scalar('train_f1_macro', train_f1_macro, global_step=epoch)
            file_writer.add_scalar('train_f1_micro', train_f1_micro, global_step=epoch)
            file_writer.add_scalar('top1_accuracy', top1_accuracy, global_step=epoch)
            file_writer.add_scalar('f1_macro', f1_macro, global_step=epoch)
            file_writer.add_scalar('f1_micro', f1_micro, global_step=epoch)

            if f1_micro > best_f1_micro:
                best_f1_micro = f1_micro
                best_epoch = epoch
            checkpoint_dir = os.path.join(TRAIN_LOG_DIR, EXPERIMENT, 'checkpoints')
            trainer.save_model(checkpoint_dir, epoch)
            val_time = time.time()-val_begin
        logging.info(
            f'Experiment: {EXPERIMENT}, epoch: {epoch}, '
            f'train time: {datetime.timedelta(seconds=int(train_time))},'
            f'val time: {datetime.timedelta(seconds=int(val_time))}'
        )

    logging.info(
        f'Experiment: {EXPERIMENT},'
        f' best epoch: {best_epoch}, '
        f'best f1 micro: {best_f1_micro}'
    )


@torch.no_grad()
def evaluate(trainer, val_dataloader):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    f1_macro = AverageMeter()
    f1_micro = AverageMeter()

    # switch to evaluate mode
    trainer.model.eval()

    end = time.time()
    for i, data in enumerate(track(val_dataloader)):
        seqs = data['seq'].cuda()
        labels = data['label'].cuda()

        args = trainer.model(seqs)
        preds = args[2]

        # measure accuracy and record loss
        prec1 = accuracy(preds, labels, topk=(1, ))
        prec1 = prec1[0]
        top1.update(prec1.item(), seqs.size(0))
        f1_macro_ = f1_score(preds, labels, 'macro')
        f1_micro_ = f1_score(preds, labels, 'micro')
        f1_macro.update(f1_macro_, seqs.size(0))
        f1_micro.update(f1_micro_, seqs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return {
        'top1': top1.avg,
        'f1_macro': f1_macro.avg,
        'f1_micro': f1_micro.avg
    }


def main(lr=0.001):
    # if args.cfg is not None:
    #     cfg.merge_from_file(args.cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in GPU_IDS])

    set_seed(RANDOM_SEED)

    log_dir = os.path.join(TRAIN_LOG_DIR, EXPERIMENT, 'logs')
    mkdir(log_dir)
    checkpoint_dir = os.path.join(TRAIN_LOG_DIR, EXPERIMENT, 'checkpoints')
    mkdir(checkpoint_dir)

    init_log(log_dir=log_dir)
    # logging.info(json.dumps(cfg, indent=4))
    file_writer = SummaryWriter(log_dir)

    train_dataset = TrainDataset(data_root=DATASET_DATA_ROOT, data_type='train')

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  shuffle=True, num_workers=4)

    val_dataset = TrainDataset(data_root=DATASET_DATA_ROOT, data_type='val')
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4)

    trainer = VAETrainer(lr)

    if TRAIN_RESUME:
        TRAIN_START_EPOCH = trainer.resume_model(TRAIN_RESUME_PATH)
    train(trainer, train_dataloader, val_dataloader, file_writer)


if __name__ == '__main__':
    learning_rate = np.float(sys.argv[1])
    main(lr=learning_rate)















