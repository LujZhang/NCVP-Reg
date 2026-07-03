from __future__ import print_function

import os

import open3d as o3d
import argparse
import torch
from numpy.ma.core import array
from scipy.spatial.distance import cdist

from scipy.spatial.transform import Rotation

from model import MultiCON
from dataset import Synthesis, pairwise_distance
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import GeodesicLoss, supervisedloss
import time
# from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from copy import deepcopy


class IOStream:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a')

    def cprint(self, text, print_flag=True):
        if print_flag:
            print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='PointCloud registration.')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum number of training epochs.', )
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.', )
    parser.add_argument('--lr', default=0.001, type=float, help='Base learning rate.', )
    parser.add_argument('--gpu_id', default=0, type=str, help='Gpu ID.')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N', help='Dropout ratio in transformer')
    args = parser.parse_args()
    return args


def compute_ir(src_pred, target_pcd, fine=False):
    dist_map = torch.sqrt(
        pairwise_distance(torch.FloatTensor(src_pred.points), torch.FloatTensor(target_pcd.points)))
    I_gt = torch.zeros_like(dist_map)
    knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
    pc_index = torch.arange(1024)
    I_gt[pc_index, knn_indices] = 1

    inner = 0
    inner_half = 0
    inner_half_half = 0
    target_pcd_points = []
    for i in range(np.asarray(src_pred.points).shape[0]):
        point = np.asarray(src_pred.points)[i]
        nearest_index = np.argmax(I_gt.squeeze(0).numpy()[i])
        nearest_point = np.asarray(target_pcd.points)[nearest_index]
        target_pcd_points.append(nearest_point)
        distance = np.linalg.norm(point - nearest_point)
        if distance < 0.01:
            inner += 1

            if fine:
                if distance < 0.005:
                    inner_half += 1
                    if distance < 0.003:
                        inner_half_half += 1

    return inner, inner_half, inner_half_half


def compute_rr(src_pcd, src_pred, target_pcd, fine=False):
    dist_map = torch.sqrt(
        pairwise_distance(torch.FloatTensor(src_pcd.points), torch.FloatTensor(target_pcd.points)))
    I_gt = torch.zeros_like(dist_map)
    knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
    pc_index = torch.arange(1024)
    I_gt[pc_index, knn_indices] = 1

    recall_inner = 0
    recall_inner_half = 0
    recall_inner_half_half = 0
    distances = []
    target_pcd_points = []
    for i in range(np.asarray(src_pred.points).shape[0]):
        point = np.asarray(src_pred.points)[i]
        nearest_index = np.argmax(I_gt.squeeze(0).numpy()[i])
        nearest_point = np.asarray(target_pcd.points)[nearest_index]
        target_pcd_points.append(nearest_point)
        distance = np.linalg.norm(point - nearest_point)


        if distance < 0.01:
            recall_inner += 1
            if fine:
                if distance < 0.005:
                    recall_inner_half += 1
                    if distance < 0.003:
                        recall_inner_half_half += 1
        distances.append(distance)

    return recall_inner, recall_inner_half, recall_inner_half_half, distances


def compute_RTE(t_gt, t_pred):
    """计算相对平移误差 (RTE), 单位：与点云坐标同单位"""
    return np.linalg.norm(t_gt - t_pred)


def chamfer_distance(P, Q):
    """计算Chamfer Distance (CD)，输出平方距离"""
    # P, Q: shape (N, 3), (M, 3)
    tree_Q = cKDTree(Q)
    dist_PQ, _ = tree_Q.query(P, k=1)  # P 到 Q 的最近邻
    tree_P = cKDTree(P)
    dist_QP, _ = tree_P.query(Q, k=1)  # Q 到 P 的最近邻
    cd = np.mean(dist_PQ ** 2) + np.mean(dist_QP ** 2)
    return cd


def compute_RRE(R_gt, R_pred):
    """计算相对旋转误差 (RRE), 单位：度"""
    R_diff = R_gt.T @ R_pred
    cos_theta = (np.trace(R_diff) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值稳定
    theta = np.arccos(cos_theta)
    RRE = np.degrees(theta)
    return RRE


def main():
    save = True
    args = parse_args()
    model = MultiCON(args).cuda(args.gpu_id)

    test_loader = DataLoader(
        Synthesis(file_path='test_b_1024.npz'),
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    model_path = './results/'
    save_path = model_path
    model_path =  'last_model.best.t7'
    model.load_state_dict(torch.load(model_path, map_location='cuda'))

    textio = IOStream(save_path + "test_log.txt")
    all_distance = []

    iter = 0
    all_inner = []
    all_inner_half = []
    all_inner_half_half = []
    all_recall_half = []
    all_recall = []
    all_recall_half_half = []
    R_all = []
    T_all = []

    gray = [0.5, 0.5, 0.5]
    # red = [1,0,1]
    src_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    ori_target_pcd = o3d.geometry.PointCloud()
    times = []

    for src, target, src_prior, tgt_prior, rotation, translation, I_gt, _ in tqdm(test_loader):


        start = time.time()
        rotation_pred, translation_pred, _, _, _= model(src.cuda(args.gpu_id), target.cuda(args.gpu_id),
                                                              src_prior.cuda(args.gpu_id), tgt_prior.cuda(args.gpu_id)) 
        end = time.time()
        times.append(end - start)

        RRE = compute_RRE(rotation.squeeze(0), rotation_pred.squeeze(0).detach().cpu().numpy())
        RTE = np.linalg.norm(translation - translation_pred.detach().cpu().numpy())
        R_all.append(RRE)
        T_all.append(RTE)

        transforms_gt = torch.eye(4)
        transforms_gt[:3, :3] = rotation.detach().cpu()
        transforms_gt[:3, 3] = translation.detach().cpu()

        transforms_pred = torch.eye(4)
        transforms_pred[:3, :3] = rotation_pred.detach().cpu()
        transforms_pred[:3, 3] = translation_pred.detach().cpu()

        ori_target_pcd.points = o3d.utility.Vector3dVector(target.squeeze(0).detach().cpu().T.numpy())

        src_pcd.points = o3d.utility.Vector3dVector(src.squeeze(0).detach().cpu().T.numpy())
        src_pred = deepcopy(src_pcd)

        src_pred.transform(transforms_pred.numpy())
        src_pcd.transform(transforms_gt.numpy())

        target_pcd.points = o3d.utility.Vector3dVector(target.squeeze(0).detach().cpu().T.numpy())

        if save:
            ori_target_pcd.paint_uniform_color([1, 0, 0])
            save_src_path = os.path.join(save_path, 'src/')
            os.makedirs(save_src_path, exist_ok=True)
            o3d.io.write_point_cloud(save_src_path + str(iter) + "_src.ply", src_pred)

            src_pred.paint_uniform_color([0, 1, 0])
            all_points = src_pred + ori_target_pcd
            save_combined_path = os.path.join(save_path, 'combined/')
            os.makedirs(save_combined_path, exist_ok=True)
            o3d.io.write_point_cloud(save_combined_path + str(iter) + "_combined_points_depth.ply", all_points)

        recall, recall_half, recall_inner_half_half, distances = compute_rr(src_pcd=src_pcd, src_pred=src_pred, target_pcd=target_pcd,
                                                    fine=True)
        inner, inner_half, inner_half_half = compute_ir(src_pred, target_pcd, fine=True)
        all_inner.append(inner / 1024)
        all_inner_half.append(inner_half / 1024)
        all_inner_half_half.append(inner_half_half / 1024)

        all_recall.append(recall / 1024)
        all_recall_half.append(recall_half / 1024)
        all_recall_half_half.append(recall_inner_half_half / 1024)
        average_distance = np.mean(distances)
        all_distance.append(average_distance)
        textio.cprint(str(iter) + "_point-----" + "distance:" + str(average_distance), print_flag=False)
        iter += 1

    mean_pair_dis = np.mean(all_distance)
    std_pair_dis = np.std(all_distance)

    mean_rre = np.mean(R_all)
    mean_rte = np.mean(T_all)

    mean_recall = np.mean(all_recall)
    mean_recall_half = np.mean(all_recall_half)
    mean_recall_half_half = np.mean(all_recall_half_half)

    mean_inner = np.mean(all_inner)
    mean_inner_half = np.mean(all_inner_half)
    mean_inner_half_half = np.mean(all_inner_half_half)

    fps = 500 / np.sum(times)
    textio.cprint('Distance: %.5f' % (mean_pair_dis))
    textio.cprint('std: %.5f' % (std_pair_dis))
    textio.cprint('RRE: %.5f' % (mean_rre))
    textio.cprint('rte: %.5f' % (mean_rte))
    textio.cprint('Recall: %.5f' % (mean_recall))
    textio.cprint('Recall 5 mm: %.5f' % (mean_recall_half))
    textio.cprint('Recall 3 mm: %.5f' % (mean_recall_half_half))
    textio.cprint('Inner ratio: %.5f' % (mean_inner))
    textio.cprint('Inner ratio 5 mm: %.5f' % (mean_inner_half))
    textio.cprint('Inner ratio 3 mm: %.5f' % (mean_inner_half_half))
    textio.cprint('FPS: %.5f' % (fps))


if __name__ == '__main__':
    main()