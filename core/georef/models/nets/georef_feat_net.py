"""
@Author: Linfang Zheng 
@Contact: zhenglinfang@icloud.com
@Time: 2024/06/06
@Note: Modified from pointnet:  https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py.
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import core.georef.models.nets.hsgcn3d as hsgcn3d

class PointMatrixNet(nn.Module):
    def __init__(self):
        super(PointMatrixNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class FeatMatrixNet(nn.Module):
    def __init__(self, dim=64):
        super(FeatMatrixNet, self).__init__()
        self.conv1 = hsgcn3d.HS_layer(dim, 64, support_num=7) 
        self.conv2 = hsgcn3d.HS_layer(64, 128, support_num=7)  
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim * dim)
        self.fc4 = nn.Linear(256, dim * dim)
        self.relu = nn.ReLU()
        self.k = dim 
        self.neighbor_num=10

    def forward(self, vertices, x):
        batchsize = x.size()[0]
        vertices = vertices.transpose(-2,-1)

        x = F.relu(self.conv1(vertices, x.transpose(-1,-2), self.neighbor_num))
        x = F.relu(self.conv2(vertices, x, self.neighbor_num))

        x = F.relu(self.conv3(x.transpose(-1,-2)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x_pre = F.relu(self.fc2(x))
        x = self.fc3(x_pre)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))) .view(1, self.k * self.k).repeat(batchsize, 1).to(x.device)
        x = x + iden
        trans_1 = x.view(-1, self.k, self.k)

        x = self.fc4(x_pre)
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))) .view(1, self.k * self.k).repeat(batchsize, 1).to(x.device)
        x = x + iden
        trans_2 = x.view(-1, self.k, self.k)
        return trans_1, trans_2
    
    
class GeoReF_FeatNet(nn.Module):
    def __init__(self, num_points, global_feat=True, out_dim=1024, **args):
        super(GeoReF_FeatNet, self).__init__()
        self.num_points = num_points
        self.out_dim = out_dim
        self.neighbor_num = 10
        self.p_mat = PointMatrixNet()
        self.hs_conv_1 = hsgcn3d.HSlayer_surface(kernel_num=64, support_num=7) 
        self.hs_conv_2 = hsgcn3d.HS_layer(64, 128, support_num=7)  
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, out_dim, 1)
        self.global_feat = global_feat
        self.f_mat = FeatMatrixNet(dim=64)

    
    def forward(self, v_pcl, v_prior, **args):
        x_r, x_ts, pcl_trans_r, pcl_trans_ts = self._extract_hs_feature(v_pcl)
        x_r_prior, x_ts_prior, prior_trans_r, prior_trans_ts = self._extract_hs_feature(v_prior)

        x_r, x_ts = self._cross_cloud_transform(x_r, x_ts, prior_trans_r, prior_trans_ts)

        feat_r, feat_ts = self._to_dense_feature(v_pcl, x_r, x_ts)
        
        feat_r_prior, feat_ts_prior = self._to_dense_feature(v_prior, x_r_prior, x_ts_prior)

        return feat_r, feat_ts, feat_r_prior, feat_ts_prior
    
    def _to_dense_feature(self, vertices, x_r, x_ts):
        x = self._extract_global_feature(vertices, x_r, x_ts)
        feat_r, feat_ts = self._to_cat_dense_feat(x, x_r, x_ts)
        return feat_r, feat_ts
    
    def _extract_hs_feature(self, vertices):
        x = self._point_deform(vertices)
        x = F.relu(self.hs_conv_1(x, self.neighbor_num)).transpose(-1,-2)
        x_r, x_ts, trans_r, trans_ts = self._feat_deform(vertices, x) 
        return x_r, x_ts, trans_r, trans_ts
    
    def _point_deform(self, x):
        trans = self.p_mat(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        return x

    def _cross_cloud_transform(self, x_r, x_ts, prior_trans_r, prior_trans_ts):
        x_r = self._mat_transform(x_r, prior_trans_r)
        x_ts = self._mat_transform(x_ts, prior_trans_ts)
        return x_r, x_ts
    
    def _extract_global_feature(self, vertices, x_r, x_ts):
        x = x_r + x_ts
        x = F.relu(self.hs_conv_2(vertices.transpose(-2,-1), x.transpose(-1,-2), self.neighbor_num)).transpose(-1,-2)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)
        return x

    def _to_cat_dense_feat(self, x, x_r, x_ts):
        n_pts = x_r.shape[2]
        x = x.view(-1, self.out_dim, 1).repeat(1, 1, n_pts)
        feat_r = torch.cat([x, x_r], 1)
        feat_ts = torch.cat([x, x_ts], 1)
        return feat_r, feat_ts
    
    def _mat_transform(self, x, trans):
        return torch.bmm(x.transpose(2,1), trans).transpose(2,1)
    
    def _feat_deform(self, vertices, x):
        trans_r, trans_ts = self.f_mat(vertices, x)
        x_ts = self._mat_transform(x, trans_ts)
        x_r = self._mat_transform(x_ts, trans_r)
        return x_r, x_ts, trans_r, trans_ts


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = PointMatrixNet()
    out = trans(sim_data)
    print("PointMatrixNet", out.size())
    print("loss", feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = FeatMatrixNet(k=64)
    out = trans(sim_data_64d)
    print("FeatMatrixNet", out.size())
    print("loss", feature_transform_regularizer(out))

    feat_net_g = GeoReF_FeatNet(global_feat=True, num_points=2500)
    out = feat_net_g(sim_data)
    print("global feat", out.size())

    feat_net = GeoReF_FeatNet(global_feat=False, num_points=2500)
    out = feat_net(sim_data)
    print("point feat", out.size())
