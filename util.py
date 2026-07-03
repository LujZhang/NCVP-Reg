import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import open3d as o3d

class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.eps_1 = 1e-16

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))
        theta = theta.mean()  # [1]

        return theta

def unsupervisedloss(src, tgt, args, alpha=1.0, beta=1.0, gamma=1.0):
    "the unsupervised loss function including three part: 1) keep distance consisitency loss; 2) keep rigid motion consisitency loss; 3) keep global consisitency loss"
    '''args:
        src,tgt: point clouds matched in order
        alpha: weight for keep distance
        beta: weight for rigid motion
        gamma: weight for keep global
    '''
    w_dis = alpha / (alpha + beta + gamma)
    w_rgd = beta / (alpha + beta + gamma)
    w_glb = gamma / (alpha + beta + gamma)
    loss1 = KDCLoss(src,tgt) # 几何形状一致性
    loss2 = KRMCLoss(src,tgt,Num=5,args=args)
    loss3 = KGCLoss(src,tgt,args) # 源经过旋转后与目标的mse
    Loss = w_dis * loss1 + w_rgd * loss2 + w_glb * loss3

    return Loss

def knn(x):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous() #b,n,n

    return pairwise_distance


def KDCLoss(src, tgt):
    dis_matrix_src = knn(src)
    dis_matrix_tgt = knn(tgt)
    mse = torch.mean((dis_matrix_src - dis_matrix_tgt) ** 2)
    return mse


def motion_esti(src, tgt, args):
    # input:
    #   scr: point cloud; 3,num_points
    #   tgt: corresponding point cloud; 3,num_points
    # output:
    #   R: Rotation matrix; 3,3
    #   t: translation matrix; 3
    # src = src.transpose(1,0)
    # tgt = tgt.transpose(1,0)
    src_centered = src - src.mean(dim=1, keepdim=True)  # 3,n
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)  # 3,n

    H = torch.matmul(src_centered, tgt_centered.transpose(1, 0).contiguous()).cpu()

    u, s, v = torch.svd(H)
    r = torch.matmul(v, u.transpose(1, 0)).contiguous()
    r_det = torch.det(r).item()
    diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                      [0, 1.0, 0],
                                      [0, 0, r_det]]).astype('float32')).to(v.device)
    r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous().cuda(args.gpu_id)
    t = torch.matmul(-r, src.mean(dim=1, keepdim=True)) + tgt.mean(dim=1, keepdim=True)

    return r, t


def KRMCLoss(src, tgt, Num, args):
    src = src.cuda(args.gpu_id)
    tgt = tgt.cuda(args.gpu_id)

    loss = 0
    batch_size = src.shape[0]   #1
    num_points = src.shape[2]   #maybe < 10


    if num_points<10:
        identity = torch.eye(3).cuda(args.gpu_id)
        src_one_batch = src[0]
        tgt_one_batch = tgt[0]  # 3,num_points
        R_global, _ = motion_esti(src_one_batch, tgt_one_batch, args)
        loss = F.mse_loss(torch.matmul(R_global, R_global), identity)
        print(num_points)
        print("<10",loss)
    elif num_points >= 10:
        n = np.floor(num_points / Num).astype(int)
        identity = torch.eye(3).cuda(args.gpu_id)
        for i in range(batch_size):
            loss_i = 0
            src_one_batch = src[i]
            tgt_one_batch = tgt[i]  # 3,num_points
            R_global, t_global = motion_esti(src_one_batch, tgt_one_batch, args)
            # R_gb, t_gb = motion_esti(src_one_batch, tgt_one_batch) #3,3;3
            s = torch.randperm(src_one_batch.shape[1])
            src_one_batch_perm = src_one_batch.transpose(1, 0)[s, :]  # 3,n -> n,3 -> ...
            tgt_one_batch_perm = tgt_one_batch.transpose(1, 0)[s, :]  # 3,n -> n,3 -> ...
            for j in range(Num):
                src_local = src_one_batch_perm[j * n:(j + 1) * n, :]  # n,3
                tgt_local = tgt_one_batch_perm[j * n:(j + 1) * n, :]  # n,3
                R_local, t_local = motion_esti(src_local.transpose(1, 0), tgt_local.transpose(1, 0), args)
                loss_j = F.mse_loss(torch.matmul(R_local.transpose(1, 0), R_global), identity) + F.mse_loss(t_local,
                                                                                                            t_global)
                loss_i = loss_i + loss_j
                loss = loss + loss_i
    else:
        # print(num_points)
        print(num_points)
        print("else:",loss)
        loss = 0.01


    return loss


def KGCLoss(src, tgt, args):
    batch_size = src.shape[0]
    R_batch = []
    t_batch = []
    for i in range(batch_size):
        R, t = motion_esti(src[i], tgt[i], args)
        R_batch.append(R)
        t_batch.append(t)
    R_batch = torch.stack(R_batch, dim=0)
    t_batch = torch.stack(t_batch, dim=0)

    # src_motion = (torch.matmul(R, src) + t.repeat(1,1,src.shape[2])) #b,3,n

    src_motion = (torch.matmul(R_batch, src) + t_batch.repeat(1, 1, src.shape[2]))  # b,3,n
    mse = torch.mean((src_motion - tgt) ** 2)
    return mse

def Myo3dKGCLoss(src, tgt, rt_pred):
    batch_size = src.shape[0]
    src_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    mse = []
    for i in range(batch_size):
        src_pcd.points = o3d.utility.Vector3dVector(src[i].detach().cpu().numpy().T)
        src_pcd.transform(rt_pred[i].detach().cpu().numpy())
        target_pcd.points = o3d.utility.Vector3dVector(tgt[i].detach().cpu().numpy().T)
        tar_tree = o3d.geometry.KDTreeFlann(target_pcd)
        distances = []
        for point in np.asarray(src_pcd.points):
            [k, idx, _] = tar_tree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(target_pcd.points)[idx[0]]
            distance = np.linalg.norm(point - nearest_point)
            distances.append(distance)

        average_distance = np.mean(distances)
        mse.append(average_distance)
    return np.mean(mse)

def compute_dynamic_k_loss(
    f_k1_src, f_k2_src,
    f_k1_tgt, f_k2_tgt,
    target_src=0.1,
    target_tgt=0.4,
    margin=0.1,
):
    """
    Dynamic k loss for partial-to-global point cloud registration
    """

    # ===== 1️⃣ src：抑制变化（局部）
    loss_src = (
        (torch.abs(f_k1_src) - target_src).pow(2).mean() +
        (torch.abs(f_k2_src) - target_src).pow(2).mean()
    )

    # ===== 2️⃣ tgt：鼓励变化（整体）
    loss_tgt = (
        (torch.abs(f_k1_tgt) - target_tgt).pow(2).mean() +
        (torch.abs(f_k2_tgt) - target_tgt).pow(2).mean()
    )

    # ===== 3️⃣ 顺序约束：src < tgt
    loss_order = (
        F.relu(torch.abs(f_k1_src) - torch.abs(f_k1_tgt) + margin).mean() +
        F.relu(torch.abs(f_k2_src) - torch.abs(f_k2_tgt) + margin).mean()
    )

    # ===== 4️⃣ 合并
    loss_k = (
        0.5 * loss_src +
        0.5 * loss_tgt +
        1.0 * loss_order
    )

    return loss_k
def compute_recall_ir_dis(src_pred, tgt, I_gt, r_pred, t_pred, threshold=0.1):
    """
    PyTorch版本的Recall、Inner Ratio、Pair Distance计算。

    Args:
        src_pred: [1, 3, N] torch.Tensor
        tgt: [1, 3, M] torch.Tensor
        I_gt: [1, N, M] torch.Tensor，GT匹配矩阵（0/1）
        r_pred: [3, 3] torch.Tensor，预测旋转矩阵
        t_pred: [3] torch.Tensor，预测平移向量
        threshold: float，距离阈值

    Returns:
        recall: float
        inner_ratio: float
        pair_distance: float
    """
    device = src_pred.device
    N = src_pred.shape[2]

    # 把src_pred变成 (N, 3)
    src_np = src_pred.squeeze(0).permute(1, 0).contiguous()  # (N, 3)
    tgt_np = tgt.squeeze(0).permute(1, 0).contiguous()  # (M, 3)
    I_gt_np = I_gt.squeeze(0)  # (N, M)


    # 变换src点云
    src_np = src_pred.squeeze(0).transpose(0, 1).contiguous()  # (N, 3)
    r_pred = r_pred.squeeze(0)  # (3, 3)
    t_pred = t_pred.squeeze(0)  # (3,)
    transformed_src = torch.matmul(src_np, r_pred.T) + t_pred.unsqueeze(0)

    # 找每个src点对应的GT匹配点索引
    gt_indices = torch.argmax(I_gt_np, dim=1)  # (N,)
    gt_match_points = tgt_np[gt_indices]  # (N, 3)

    # 计算每对对应点的距离
    pair_distances = torch.norm(transformed_src - gt_match_points, dim=1)  # (N,)

    # recall，距离小于阈值的比例
    recall = (pair_distances < threshold).float().mean().item()

    # 平均配对距离
    pair_distance = pair_distances.mean().item()

    # 计算每个变换后src点到tgt所有点的距离
    # 利用广播，计算距离矩阵 (N, M)
    diff = transformed_src.unsqueeze(1) - tgt_np.unsqueeze(0)  # (N, M, 3)
    dists = torch.norm(diff, dim=2)  # (N, M)
    nearest_distances, _ = torch.min(dists, dim=1)  # (N,)

    # inner ratio，距离最近的点小于阈值的比例
    inner_ratio = (nearest_distances < threshold).float().mean().item()

    return recall, inner_ratio, pair_distance

def Mysupervisedloss(gt, rt_pred):
    loss = torch.mean(torch.square(rt_pred - gt))
    return loss

def MyKGCLoss(src, tgt, rotation, translation):
    src_motion = (torch.matmul(rotation, src) + translation.unsqueeze(2).repeat(1, 1, src.shape[2]))
    mse = torch.mean((src_motion - tgt) ** 2)
    return mse

def supervisedloss(I_gt, I_pre, args):
    #input:
    #   pre: prediction matching matrix ; b,M,N
    #   gt:  ground truth matching matrix; b,M,N
    #log_pre = torch.log(pre)
    I_gt = I_gt
    I_pre = I_pre
    # loss=torch.sum((I_gt-I_pre)**2)
    # loss=torch.log(loss)
    loss_up = torch.sum(torch.mul(I_gt,I_pre))
    loss_down = torch.sum(I_gt)
    loss = -1.0*loss_up/loss_down
    # print(loss)
    #print("matching loss: {}".format(loss))

    return loss

