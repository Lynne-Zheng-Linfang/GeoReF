import numpy as np
from lib.pysixd.pose_error import re, te
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import rot6d_to_mat_batch
from core.utils import lie_algebra, quaternion_lf


def get_rot_dim(rot_type):
    if rot_type in ["allo_quat", "ego_quat"]:
        rot_dim = 4
    elif rot_type in [
        "allo_log_quat",
        "ego_log_quat",
        "allo_lie_vec",
        "ego_lie_vec",
    ]:
        rot_dim = 3
    elif rot_type in ["allo_rot6d", "ego_rot6d"]:
        rot_dim = 6
    else:
        raise ValueError(f"Unknown rot_type: {rot_type}")
    return rot_dim


def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_log_quat", "allo_log_quat"]:
        # from latentfusion (lf)
        rot_m = quat2mat_torch(quaternion_lf.qexp(rot))
    elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
        rot_m = lie_algebra.lie_vec_to_rot(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m

def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()
