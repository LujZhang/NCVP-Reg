import numpy as np
import torch
import torch.utils.data
import open3d as o3d
import torch.nn as nn


def compute_bounding_box_info(point_cloud):

    if isinstance(point_cloud, np.ndarray):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(point_cloud)
        point_cloud = pc

    # 计算包围框信息
    bbox = point_cloud.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    extent = bbox.get_extent()
    center = bbox.get_center()

    # 拼接为向量
    bbox_vector = np.concatenate([min_bound, max_bound, extent, center])
    bbox_vector = torch.FloatTensor(bbox_vector).unsqueeze(0)  # (1, 12)

    return bbox_vector

class Synthesis(torch.utils.data.Dataset):
    def __init__(self, file_path, transformations=None):
        self.file_path = file_path
        self.transformations = transformations
        self.num_point = 1024

        array = np.load(self.file_path)
        self.pc = array['PC']
        self.length = len(self.pc)

        self.ct_array = array['CT']
        self.gt_array = array['GT']

        # self.landmark_array = array['Landmark']

    def __getitem__(self, index):

        pc = self.pc[index].astype(np.float32)
        ct = self.ct_array[int(pc[0,3])-1] / 1000 # 点云每个点的6个位置存储着该患者的编号，ct:20480 * 3,单位转为米

        gt = self.gt_array[index].astype(np.float32)
        gt = np.linalg.inv(gt)

        pointcloud_pc = o3d.geometry.PointCloud()
        pointcloud_pc.points = o3d.utility.Vector3dVector(pc[:, :3])

        src_prior = compute_bounding_box_info(pointcloud_pc)
        tgt_prior = compute_bounding_box_info(ct)


        pointcloud_pc.transform(gt)
        dist_map = torch.sqrt(pairwise_distance(torch.FloatTensor(pointcloud_pc.points), torch.FloatTensor(ct)))
        Ig = torch.zeros_like(dist_map)
        knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
        pc_index = torch.arange(1024)
        Ig[pc_index, knn_indices] = 1

        overlap_count = 0
        kdtree = o3d.geometry.KDTreeFlann(pointcloud_pc)
        for point in ct:
            [_, idx, _] = kdtree.search_radius_vector_3d(point, radius=0.004)
            if len(idx) > 0:
                overlap_count += 1
        overlap_ratio = overlap_count / 1024

        # pointcloud = o3d.geometry.PointCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(ct)
        # pointcloud_pc = o3d.geometry.PointCloud()
        # pointcloud_pc.points = o3d.utility.Vector3dVector(pc[:, :3])
        # pointcloud_pc.transform(gt)
        # o3d.visualization.draw_geometries([pointcloud, pointcloud_pc])

        # landmark = self.landmark_array[index].astype(np.float32) / 1000

        return (torch.FloatTensor(pc[:, :3].T), torch.FloatTensor(ct.T), src_prior, tgt_prior,
                torch.FloatTensor(gt[:3, :3]), torch.FloatTensor(gt[:3, 3]), Ig, overlap_ratio)

    def __len__(self):
        return self.length

def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances