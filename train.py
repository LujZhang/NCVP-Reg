#!/usr/bin/env python
import os
import argparse
import torch
import torch.optim as optim
from model import MultiCON
from util import unsupervisedloss, Mysupervisedloss, supervisedloss
from dataset import Synthesis
from torchinfo import summary

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='PointCloud registration.')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum number of training epochs.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='Base learning rate.')
    parser.add_argument('--gpu_id', default=0, type=str, help='Gpu ID.')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N', help='Dropout ratio in transformer')
    args = parser.parse_args()
    return args


class IOStream:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_():
    if not os.path.exists('results'):
        os.makedirs('results')


def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    _init_()
    exp_name = 'non_local_v2_wo_prior'
    textio = IOStream('results/' + str(exp_name) + '/run.log')

    train_loader = DataLoader(
        Synthesis('./train_b_1024.npz'),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        Synthesis('./test_b_1024.npz'),
        batch_size=args.batch_size, shuffle=False, drop_last=False)

    net = MultiCON(args).cuda(args.gpu_id)
    input_size = [(1, 3, 1024), (1, 3, 1024), (1, 1, 12), (1, 1, 12)]
    summary(net, input_size=input_size)
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)

    best_test_loss = np.inf

    for epoch in range(args.epochs):
        scheduler.step()
        (train_loss, train_unsupervised_loss, train_rot_supervised_loss,
         train_trans_supervised_loss, train_matching_losses) = train_one_epoch(
            args, net, train_loader, opt)

        with torch.no_grad():
            (test_loss, test_unsupervised_loss, test_rot_supervised_loss,
             test_trans_supervised_loss, test_matching_losses) = test_one_epoch(
                args, net, test_loader)

        train_unsupervised_mean = np.mean(train_unsupervised_loss)
        train_rot_supervised_mean = np.mean(train_rot_supervised_loss)
        train_trans_supervised_mean = np.mean(train_trans_supervised_loss)
        train_matching_mean = np.mean(train_matching_losses)

        test_unsupervised_mean = np.mean(test_unsupervised_loss)
        test_rot_supervised_mean = np.mean(test_rot_supervised_loss)
        test_trans_supervised_mean = np.mean(test_trans_supervised_loss)
        test_matching_mean = np.mean(test_matching_losses)

        torch.save(net.state_dict(), 'results/' + str(exp_name) + '/last_model.best.t7')

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_unsupervised_mean = test_unsupervised_mean
            best_rot_supervised_mean = test_rot_supervised_mean
            best_trans_supervised_mean = test_trans_supervised_mean
            best_matching_mean = test_matching_mean
            torch.save(net.state_dict(), 'results/' + str(exp_name) + '/best_model.best.t7')

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, unsupervised: %f, rot_supervised: %f, '
                      'trans_supervised: %f, matching: %f'
                      % (epoch, train_loss, train_unsupervised_mean, train_rot_supervised_mean,
                         train_trans_supervised_mean, train_matching_mean))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, unsupervised: %f, rot_supervised: %f, '
                      'trans_supervised: %f, matching: %f'
                      % (epoch, test_loss, test_unsupervised_mean, test_rot_supervised_mean,
                         test_trans_supervised_mean, test_matching_mean))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, unsupervised: %f, rot_supervised: %f, '
                      'trans_supervised: %f, matching: %f'
                      % (epoch, best_test_loss, best_unsupervised_mean, best_rot_supervised_mean,
                         best_trans_supervised_mean, best_matching_mean))


def test_one_epoch(args, net, test_loader):
    total_loss = 0
    num_samples = 0
    unsupervised_losses = []
    rot_supervised_losses = []
    trans_supervised_losses = []
    matching_losses = []

    for src, target, src_prior, tgt_prior, rotation, translation, Ig, _ in tqdm(test_loader):
        src = src.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        src_prior = src_prior.cuda(args.gpu_id)
        tgt_prior = tgt_prior.cuda(args.gpu_id)
        rotation = rotation.cuda(args.gpu_id)
        translation = translation.cuda(args.gpu_id)
        Ig = Ig.cuda(args.gpu_id)

        batch_size = src.size(0)
        num_samples += batch_size
        rotation_pred, translation_pred, src, corr_src, scores = net(src, target, src_prior, tgt_prior)

        unsupervised_loss_val = unsupervisedloss(src, corr_src, args)
        rot_supervised_loss_val = Mysupervisedloss(rotation_pred, rotation)
        trans_supervised_loss_val = Mysupervisedloss(translation_pred, translation)

        lambda_supervised = 100.0
        supervised_loss_val = supervisedloss(Ig, scores, args)
        supervised_loss_val = lambda_supervised * supervised_loss_val
        loss = unsupervised_loss_val + supervised_loss_val

        unsupervised_losses.append(unsupervised_loss_val.detach().cpu().numpy())
        rot_supervised_losses.append(rot_supervised_loss_val.detach().cpu().numpy())
        trans_supervised_losses.append(trans_supervised_loss_val.detach().cpu().numpy())
        matching_losses.append(supervised_loss_val.detach().cpu().numpy())

        total_loss += loss.item() * batch_size

    return (total_loss * 1.0 / num_samples,
            unsupervised_losses, rot_supervised_losses,
            trans_supervised_losses, matching_losses)


def train_one_epoch(args, net, train_loader, opt):
    net.train()
    num_samples = 0
    total_loss = 0
    unsupervised_losses = []
    rot_supervised_losses = []
    trans_supervised_losses = []
    matching_losses = []

    for src, target, src_prior, tgt_prior, rotation, translation, Ig, _ in tqdm(train_loader):
        src = src.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        src_prior = src_prior.cuda(args.gpu_id)
        tgt_prior = tgt_prior.cuda(args.gpu_id)
        rotation = rotation.cuda(args.gpu_id)
        translation = translation.cuda(args.gpu_id)
        Ig = Ig.cuda(args.gpu_id)

        batch_size = src.size(0)
        opt.zero_grad()
        num_samples += batch_size
        rotation_pred, translation_pred, src, corr_src, scores = net(src, target, src_prior, tgt_prior)

        unsupervised_loss_val = unsupervisedloss(src, corr_src, args)
        rot_supervised_loss_val = Mysupervisedloss(rotation_pred, rotation)
        trans_supervised_loss_val = Mysupervisedloss(translation_pred, translation)

        lambda_supervised = 100.0
        supervised_loss_val = supervisedloss(Ig, scores, args)
        supervised_loss_val = lambda_supervised * supervised_loss_val
        loss = unsupervised_loss_val + supervised_loss_val

        unsupervised_losses.append(unsupervised_loss_val.detach().cpu().numpy())
        rot_supervised_losses.append(rot_supervised_loss_val.detach().cpu().numpy())
        trans_supervised_losses.append(trans_supervised_loss_val.detach().cpu().numpy())
        matching_losses.append(supervised_loss_val.detach().cpu().numpy())

        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size

    return (total_loss * 1.0 / num_samples,
            unsupervised_losses, rot_supervised_losses,
            trans_supervised_losses, matching_losses)


if __name__ == '__main__':
    main()
