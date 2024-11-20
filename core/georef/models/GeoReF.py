import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.events import get_event_storage
from core.utils.solver_utils import build_optimizer_with_params

from core.utils.my_checkpoint import load_timm_pretrained
from mmcv.runner import load_checkpoint

from .nets.georef_feat_net import GeoReF_FeatNet
from .heads.fc_trans_size_head import FC_TransSizeHead
from .heads.conv_out_per_rot_head import ConvOutPerRotHead
from .model_utils import (
    compute_mean_re_te,
    get_rot_mat,
)

from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss

from .pose_scale_from_delta_init import pose_scale_from_delta_init

logger = logging.getLogger(__name__)


class GeoReF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL.GeoReF
        self.params_lr_list = []
        self.feat_net = self._get_feat_net(cfg)
        self.rot_head = self._get_rot_net(cfg)
        self.ts_head = self._get_ts_head(cfg) 

    def _get_rot_net(self, cfg):
        rot_head_cfg = self.model_cfg.ROT_HEAD
        rot_num_classes = self.model_cfg.NUM_CLASSES if rot_head_cfg.CLASS_AWARE else 1

        rot_head_init_cfg = copy.deepcopy(rot_head_cfg.INIT_CFG)
        rot_head_init_cfg.update(num_classes=rot_num_classes)

        rot_head = ConvOutPerRotHead(**rot_head_init_cfg)
        self.update_params_lr_list(rot_head, rot_head_cfg.FREEZE, cfg.SOLVER.BASE_LR, rot_head_cfg.LR_MULT)
        return rot_head

    def update_params_lr_list(self, net, freeze, base_lr, lr_mult=1.0):
        if freeze:
            for param in net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            assert self.params_lr_list is not None
            new_params = {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": float(base_lr)*lr_mult}
            self.params_lr_list.append(new_params)

    def _get_feat_net(self, cfg):
        feat_net_cfg = self.model_cfg.FEATNET
        init_feat_net_args = copy.deepcopy(feat_net_cfg.INIT_CFG)
        feat_net = GeoReF_FeatNet(**init_feat_net_args)
        self.update_params_lr_list(feat_net, feat_net_cfg.FREEZE, cfg.SOLVER.BASE_LR)
        return feat_net
    
    def _get_ts_head(self, cfg):
        ts_head_cfg = self.model_cfg.TS_HEAD
        num_classes = self.model_cfg.NUM_CLASSES if self.model_cfg.ROT_HEAD.CLASS_AWARE else 1

        ts_head_init_cfg = copy.deepcopy(ts_head_cfg.INIT_CFG)
        ts_head_init_cfg.update(num_classes=num_classes)
        ts_head = FC_TransSizeHead(**ts_head_init_cfg)
        self.update_params_lr_list(ts_head, ts_head_cfg.FREEZE, cfg.SOLVER.BASE_LR, ts_head_cfg.LR_MULT)
        return ts_head
    
    def forward(
        self,
        x,
        tfd_kps,
        init_pose,
        init_scale,
        K_zoom=None,
        obj_class=None,
        gt_ego_rot=None,
        gt_trans=None,
        gt_scale=None,
        obj_kps=None,
        mean_scales=None,
        sym_info=None,
        do_loss=False,
        cur_iter=0,
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.GeoReF
        rot_head_cfg = net_cfg.ROT_HEAD
        ts_head_cfg = net_cfg.TS_HEAD

        num_classes = net_cfg.NUM_CLASSES
        device = x.device
        bs = x.shape[0]

        pcl_feat_r, pcl_feat_ts, kps_feat_r, kps_feat_ts= self.feat_net(x, tfd_kps)  # [bs, c, num_p]

        rot_feat = torch.cat((pcl_feat_r, kps_feat_r), dim=2)  # bs, c, num_pcl+num_kps
        n_pcl = pcl_feat_r.size(-1)
        rot_deltas_ = self.rot_head(rot_feat)

        # use scale as explicit input
        if ts_head_cfg.WITH_INIT_SCALE:
            t_feat = torch.cat((pcl_feat_ts, kps_feat_ts), dim=2)  # bs, c, num_pcl+num_kps
            s_feat = torch.cat([t_feat, init_scale.unsqueeze(-1).repeat(1,1,t_feat.size(-1))], dim=1)  # [bs, c'+3]
        trans_deltas_, scale_deltas_ = self.ts_head(t_feat, s_feat)

        if rot_head_cfg.CLASS_AWARE:
            assert obj_class is not None
            rot_deltas_ = rot_deltas_.view(bs, num_classes, self.pose_head.rot_dim)
            rot_deltas_ = rot_deltas_[torch.arange(bs).to(device), obj_class]
            trans_deltas_ = trans_deltas_.view(bs, num_classes, 3)
            trans_deltas_ = trans_deltas_[torch.arange(bs).to(device), obj_class]

        # convert pred_rot to rot mat -------------------------
        rot_m_deltas = get_rot_mat(rot_deltas_, rot_type=rot_head_cfg.ROT_TYPE)
        # rot_m_deltas, trans_deltas, init_pose --> ego pose -----------------------------
        pred_ego_rot, pred_trans, pred_scale = pose_scale_from_delta_init(
            rot_deltas=rot_m_deltas,
            trans_deltas=trans_deltas_,
            scale_deltas=scale_deltas_,
            rot_inits=init_pose[:, :3, :3],
            trans_inits=init_pose[:, :3, 3],
            scale_inits=init_scale if "iter" in rot_head_cfg.SCLAE_TYPE else mean_scales,
            Ks=K_zoom,  # Ks without zoom # no need
            K_aware=rot_head_cfg.T_TRANSFORM_K_AWARE,
            delta_T_space=rot_head_cfg.DELTA_T_SPACE,
            delta_T_weight=rot_head_cfg.DELTA_T_WEIGHT,
            delta_z_style=rot_head_cfg.DELTA_Z_STYLE,
            eps=1e-4,
            is_allo="allo" in rot_head_cfg.ROT_TYPE,
            scale_type=rot_head_cfg.SCLAE_TYPE,
        )
        pred_pose = torch.cat([pred_ego_rot, pred_trans.view(-1, 3, 1)], dim=-1)

        # NOTE: an ablation setting
        if not cfg.MODEL.REFINE_SCLAE:
            pred_scale = init_scale

        out_dict = {f"pose_{cur_iter}": pred_pose, f"scale_{cur_iter}": pred_scale}
        if not do_loss:  # test
            return out_dict
        else:
            assert gt_ego_rot is not None and (gt_trans is not None)
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_ego_rot, gt_trans, gt_ego_rot)
            # yapf: disable
            vis_dict = {
                f"vis/error_R_{cur_iter}": mean_re,  # deg
                f"vis/error_t_{cur_iter}": mean_te * 100,  # cm
                f"vis/error_tx_{cur_iter}": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                f"vis/error_ty_{cur_iter}": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                f"vis/error_tz_{cur_iter}": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                f"vis/tx_pred_{cur_iter}": pred_trans[0, 0].detach().item(),
                f"vis/ty_pred_{cur_iter}": pred_trans[0, 1].detach().item(),
                f"vis/tz_pred_{cur_iter}": pred_trans[0, 2].detach().item(),
                f"vis/tx_delta_{cur_iter}": trans_deltas_[0, 0].detach().item(),
                f"vis/ty_delta_{cur_iter}": trans_deltas_[0, 1].detach().item(),
                f"vis/tz_delta_{cur_iter}": trans_deltas_[0, 2].detach().item(),
                f"vis/tx_gt_{cur_iter}": gt_trans[0, 0].detach().item(),
                f"vis/ty_gt_{cur_iter}": gt_trans[0, 1].detach().item(),
                f"vis/tz_gt_{cur_iter}": gt_trans[0, 2].detach().item(),
            }

            loss_dict = self.georef_loss(
                out_rot=pred_ego_rot, gt_rot=gt_ego_rot,
                out_trans=pred_trans, gt_trans=gt_trans,
                out_scale=pred_scale, gt_scale=gt_scale,
                obj_kps=obj_kps, sym_info=sym_info,
            )

            if net_cfg.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}_{cur_iter}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            # yapf: enable
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict

    def georef_loss(
        self,
        out_rot,
        out_trans,
        out_scale,
        gt_rot=None,
        gt_trans=None,
        gt_scale=None,
        obj_kps=None,
        sym_info=None,
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.GeoReF
        loss_cfg = net_cfg.LOSS_CFG

        loss_dict = {}
        # point matching loss ---------------
        if loss_cfg.PM_LW > 0:
            assert (obj_kps is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=loss_cfg.PM_LOSS_TYPE,
                beta=loss_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=loss_cfg.PM_LW,
                symmetric=loss_cfg.PM_LOSS_SYM,
                disentangle_t=loss_cfg.PM_DISENTANGLE_T,
                disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
                r_only=loss_cfg.PM_R_ONLY,
                with_scale=loss_cfg.PM_WITH_SCALE,
                use_bbox=loss_cfg.PM_USE_BBOX,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=obj_kps,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                pred_scales=out_scale,
                gt_scales=gt_scale,
                sym_infos=sym_info,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss (symmetry-aware) ----------
        if loss_cfg.ROT_LW > 0:
            # NOTE: for now all sym_infos are about y axis. If new sym_type is introduced, please change the code here.
            sym_mask = torch.tensor([0 if sym is None else 1 for sym in sym_info]).to(out_rot.device)
            out_rot_nosym = torch.index_select(out_rot, dim=0, index=torch.where(sym_mask == 0)[0])
            gt_rot_nosym = torch.index_select(gt_rot, dim=0, index=torch.where(sym_mask == 0)[0])
            out_rot_sym = torch.index_select(out_rot, dim=0, index=torch.where(sym_mask == 1)[0])
            gt_rot_sym = torch.index_select(gt_rot, dim=0, index=torch.where(sym_mask == 1)[0])

            # for non-sym object
            if out_rot_nosym.shape[0] > 0:
                if loss_cfg.ROT_LOSS_TYPE == "angular":
                    loss_dict["loss_rot"] = angular_distance(out_rot_nosym, gt_rot_nosym)
                elif loss_cfg.ROT_LOSS_TYPE == "L2":
                    loss_dict["loss_rot"] = rot_l2_loss(out_rot_nosym, gt_rot_nosym)
                else:
                    raise ValueError(f"Unknown rot loss type: {loss_cfg.ROT_LOSS_TYPE}")
                loss_dict["loss_rot"] *= loss_cfg.ROT_LW

            # for sym object, just the second column
            if out_rot_sym.shape[0] > 0:
                if loss_cfg.ROT_YAXIS_LOSS_TYPE == "L1":
                    loss_dict["loss_yaxis_rot"] = nn.L1Loss(reduction="mean")(out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1])
                elif loss_cfg.ROT_YAXIS_LOSS_TYPE == "smoothL1":
                    loss_dict["loss_yaxis_rot"] = nn.SmoothL1Loss(reduction="mean")(
                        out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1]
                    )
                elif loss_cfg.ROT_YAXIS_LOSS_TYPE == "L2":
                    loss_dict["loss_yaxis_rot"] = L2Loss(reduction="mean")(out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1])
                elif loss_cfg.ROT_YAXIS_LOSS_TYPE == "angular":
                    loss_dict["loss_yaxis_rot"] = angular_distance(out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1])
                else:
                    raise ValueError(f"Unknown rot yaxis loss type: {loss_cfg.ROT_YAXIS_LOSS_TYPE}")
                loss_dict["loss_yaxis_rot"] *= loss_cfg.ROT_LW

        # trans loss ------------------
        if loss_cfg.TRANS_LW > 0:
            if loss_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= loss_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= loss_cfg.TRANS_LW
            else:
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= loss_cfg.TRANS_LW

        # scale loss ---------------------
        if loss_cfg.SCALE_LW > 0:
            assert cfg.MODEL.REFINE_SCLAE
            if loss_cfg.SCALE_LOSS_TYPE == "L1":
                loss_dict["loss_scale"] = nn.L1Loss(reduction="mean")(out_scale, gt_scale)
            elif loss_cfg.SCALE_LOSS_TYPE == "L2":
                loss_dict["loss_scale"] = L2Loss(reduction="mean")(out_scale, gt_scale)
            elif loss_cfg.SCALE_LOSS_TYPE == "MSE":
                loss_dict["loss_scale"] = nn.MSELoss(reduction="mean")(out_scale, gt_scale)
            else:
                raise ValueError(f"Unknown scale loss type: {loss_cfg.SCALE_LOSS_TYPE}")
            loss_dict["loss_scale"] *= loss_cfg.SCALE_LW

        return loss_dict

def init_weights(cfg, model):
    feat_net_cfg = cfg.MODEL.GeoReF.FEATNET
    if cfg.MODEL.WEIGHTS == "":
        ## feat_net initialization
        feat_net_pretrained = feat_net_cfg.get("PRETRAINED", "")
        if feat_net_pretrained == "":
            logger.warning("Randomly initialize weights for feat_net!")
        elif feat_net_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info("Check if the feat_net has been initialized with its own method!")
            if feat_net_pretrained == "timm":
                if feat_net_cfg.INIT_CFG.pretrained and feat_net_cfg.INIT_CFG.in_chans != 3:
                    load_timm_pretrained(
                        model.feat_net, in_chans=feat_net_cfg.INIT_CFG.in_chans, adapt_input_mode="custom", strict=False
                    )
                    logger.warning("override input conv weight adaptation of timm")
        else:
            # initialize feat_net with official weights
            tic = time.time()
            logger.info(f"load feat_net weights from: {feat_net_pretrained}")
            load_checkpoint(model.feat_net, feat_net_pretrained, strict=False, logger=logger)
            logger.info(f"load feat_net weights took: {time.time() - tic}s")
    return model


def build_model_optimizer(cfg, is_test=False):
    # ================================================
    # build model
    model = GeoReF(cfg)
    optimizer = None if is_test else build_optimizer_with_params(cfg, model.params_lr_list)
    model = init_weights(cfg, model)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
