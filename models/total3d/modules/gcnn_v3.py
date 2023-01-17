# SGCN
# author: chengzhang

import torch
import torch.nn as nn
from models.registers import MODULES
from configs.data_config import NYU40CLASSES, pix3d_n_classes
import torch.nn.functional as F
from net_utils.libs import get_bdb_form_from_corners, recover_points_to_world_sys, \
    get_rotation_matix_result, get_bdb_3d_result, recover_points_to_obj_sys

from torchvision import transforms
def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class _Collection_Unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        normal_init(self.fc, 0, 0.01)

    def forward(self, target, source, attention_base):
        # assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)  # Nobj x Nrel Nrel x dim
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-7)
        return collect_avg


class _Update_Unit(nn.Module):
    def __init__(self, dim):
        super(_Update_Unit, self).__init__()
    def forward(self, target, source):
        assert target.size() == source.size(), "source dimension must be equal to target dimension"
        update = target + source
        return update


class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (subject) from rel
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (object) from rel
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (subject)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (object)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_obj)) # obj from obj

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(dim_obj)) # obj from others
        self.update_units.append(_Update_Unit(dim_rel)) # rel from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h

class GloRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, ConvNd=nn.Conv2d, BatchNormNd=nn.BatchNorm2d):
        super(GloRe_Unit, self).__init__()
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04)  # should be zero initialized

    def forward(self, x):
        '''
                :param x: (n, c, d, h, w)
                '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))

        return out

class RelReasoning(nn.Module):
    def __init__(self, cls=len(NYU40CLASSES)):
        super(RelReasoning, self).__init__()
        self.glore1 = GloRe_Unit(128, 64)
        self.glore2 = GloRe_Unit(256, 64)
        self.glore3 = GloRe_Unit(512, 64)

        self.ch_re1 = nn.Sequential(
            nn.AvgPool2d(32),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.2)
        )

        self.ch_re2 = nn.Sequential(
            nn.AvgPool2d(16),
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.2)
        )

        self.ch_re3 = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(512, 128, 1),
            nn.LeakyReLU(0.2)
        )

        self.classifier = nn.Linear(128, cls)

        self.relu_1 = nn.LeakyReLU(0.2)

        self.drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.maskfilter = transforms.GaussianBlur(7)

    def mask(self, size, bbox):
        masked = torch.zeros((1, size[0], size[1])).to(bbox.get_device())
        masked[:, int(bbox[0] / 256 * size[0]):int(bbox[2] / 256 * size[0]), int(bbox[1] / 256 * size[1]):int(bbox[3] / 256 * size[1])]=1

        return self.maskfilter(masked) +masked

    def forward(self, res_feat, bbox_pos, split):
        res1, res2, res3 = res_feat

        rel1 = self.glore1(res1)
        rel2 = self.glore2(res2)
        rel3 = self.glore3(res3)

        rel_emb1 = []
        rel_emb2 = []
        rel_emb3 = []

        for batch, s in enumerate(split):
            for i in range(s[0], s[1]):
                mask1 = self.mask(res1.shape[2:], bbox_pos[i])
                rel_emb1.append((rel1[batch] * mask1).unsqueeze(0))

                mask2 = self.mask(res2.shape[2:], bbox_pos[i])
                rel_emb2.append((rel2[batch] * mask2).unsqueeze(0))

                mask3 = self.mask(res3.shape[2:], bbox_pos[i])
                rel_emb3.append((rel3[batch] * mask3).unsqueeze(0))

        rel_feat = (self.ch_re1(torch.cat(rel_emb1)) + self.ch_re2(torch.cat(rel_emb2)) + self.ch_re3(torch.cat(rel_emb3)))/3
        rel_feat = rel_feat.squeeze(-1).squeeze(-1)

        cls = self.drop(rel_feat)
        cls = self.classifier(cls)
        cls_score = self.softmax(cls)

        return rel_feat, cls_score

@MODULES.register_module
class GCNN3(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(GCNN3, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''configs and params'''
        self.cfg = cfg
        self.lo_features = cfg.config['model']['output_adjust']['lo_features']
        self.obj_features = cfg.config['model']['output_adjust']['obj_features']
        self.rel_features = cfg.config['model']['output_adjust']['rel_features']
        feature_dim = cfg.config['model']['output_adjust']['feature_dim']
        self.feat_update_step = cfg.config['model']['output_adjust']['feat_update_step']
        self.res_output = cfg.config['model']['output_adjust'].get('res_output', False)
        self.feat_update_group = cfg.config['model']['output_adjust'].get('feat_update_group', 1)
        self.res_group = cfg.config['model']['output_adjust'].get('res_group', False)

        self.feature_length = {
            'size_cls': len(NYU40CLASSES), 'cls_codes': pix3d_n_classes,
            'bdb2D_pos': 4, 'g_features': 32, 'mgn_afeature': 1024, 'K': 3,
            'pitch_reg_result': 2, 'roll_reg_result': 2, 'pitch_cls_result': 2,
            'roll_cls_result': 2, 'lo_ori_reg_result': 2, 'lo_ori_cls_result': 2,
            'lo_centroid_result': 3, 'lo_coeffs_result': 3, 'lo_afeatures': 2048,
            'size_reg_result': 3, 'ori_reg_result': 6, 'ori_cls_result': 6,
            'centroid_reg_result': 6, 'centroid_cls_result': 6, 'offset_2D_result': 2,
            'odn_afeature': 2048, 'odn_rfeatures': 2048, 'odn_arfeatures': 2048,
            # 'ldif_afeature': cfg.config['model']['mesh_reconstruction'].get('bottleneck_size', None),
            # 'analytic_code': cfg.config['model']['mesh_reconstruction'].get('analytic_code_len', None),
            # 'blob_center': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0)
            #                + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) * 3,
            # 'ldif_phy': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0)
            #                + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) // 2,
            # 'structured_implicit_vector': cfg.config['model']['mesh_reconstruction'].get('structured_implicit_vector_len', None)
        }

        bin = cfg.dataset_config.bins
        self.OBJ_ORI_BIN = len(bin['ori_bin'])
        self.OBJ_CENTER_BIN = len(bin['centroid_bin'])
        self.PITCH_BIN = len(bin['pitch_bin'])
        self.ROLL_BIN = len(bin['roll_bin'])
        self.LO_ORI_BIN = len(bin['layout_ori_bin'])


        self.rel = RelReasoning()

        # feature to output (from Total3D object_detection)
        # branch to predict the size
        odn_dim = 128
        lo_dim = 2048
        cam_dim = 1024
        rel_dim = 128
        self.fc1 = nn.Linear(odn_dim+rel_dim, odn_dim)
        self.fc2 = nn.Linear(odn_dim, 3)

        # branch to predict the orientation
        self.fc3 = nn.Linear(odn_dim+rel_dim, odn_dim)
        self.fc4 = nn.Linear(odn_dim, self.OBJ_ORI_BIN * 2)

        # branch to predict the centroid
        self.fc5 = nn.Linear(odn_dim+rel_dim, odn_dim)
        self.fc_centroid = nn.Linear(odn_dim, self.OBJ_CENTER_BIN * 2)

        # branch to predict the 2D offset
        self.fc_off_1 = nn.Linear(odn_dim+rel_dim, odn_dim)
        self.fc_off_2 = nn.Linear(odn_dim, 2)

        # feature to output (from Total3D layout_estimation)
        self.fc_1 = nn.Linear(cam_dim, cam_dim//2)
        self.fc_2 = nn.Linear(cam_dim//2, (self.PITCH_BIN + self.ROLL_BIN) * 2)
        # fc for layout
        self.fc_layout = nn.Linear(lo_dim+cam_dim, lo_dim)
        # for layout orientation
        self.fc_3 = nn.Linear(lo_dim, lo_dim // 2)
        self.fc_4 = nn.Linear(lo_dim // 2, self.LO_ORI_BIN * 2)
        # for layout centroid and coefficients
        self.fc_5 = nn.Linear(lo_dim, lo_dim // 2)
        self.fc_6 = nn.Linear(lo_dim // 2, 6)

        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _K2feature(self, K):
        camKs = K.reshape(K.shape[0], -1)
        camKs = camKs.index_select(1, torch.tensor([0, 2, 4, 5], device=camKs.device))
        camKs = camKs[:, :3] / camKs[:, 3:]
        return camKs

    def _get_bdb3D_form(self, data):
        # camera orientation for evaluation
        cam_R_out = get_rotation_matix_result(self.cfg.bins_tensor,
                                              torch.argmax(data['pitch_cls_result'], 1),
                                              data['pitch_reg_result'],
                                              torch.argmax(data['roll_cls_result'], 1),
                                              data['roll_reg_result'])

        # projected center
        P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 - (
                data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * data['offset_2D_result'][:, 0],
                                (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 - (
                                        data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * data['offset_2D_result'][:, 1]), 1)

        # retrieved 3D bounding box
        bdb3D_result, _ = get_bdb_3d_result(self.cfg.bins_tensor,
                                            torch.argmax(data['ori_cls_result'], 1),
                                            data['ori_reg_result'],
                                            torch.argmax(data['centroid_cls_result'], 1),
                                            data['centroid_reg_result'],
                                            data['size_cls'],
                                            data['size_reg_result'],
                                            P_result,
                                            data['K'],
                                            cam_R_out,
                                            data['split'])
        bdb3D_form = get_bdb_form_from_corners(bdb3D_result)

        return bdb3D_form

    def _get_object_features(self, data, type):
        features = []
        keys = self.obj_features if type == 'obj' else self.rel_features
        for k in keys:
            if k in ['size_cls', 'cls_codes', 'size_reg_result', 'ori_reg_result', 'ori_cls_result',
                     'centroid_reg_result', 'centroid_cls_result', 'offset_2D_result',
                     'ldif_afeature', 'mgn_afeature', 'odn_afeature', 'odn_rfeatures', 'odn_arfeatures']:
                v = data[k]
            elif k == 'g_features':
                assert type == 'rel'
                v = data[k]
            elif k == 'bdb2D_pos':
                v = data[k].clone()
                center_inds = data['K'][:, :2, 2]
                for center_ind, (start, end) in zip(center_inds, data['split']):
                    for i in range(start.item(), end.item()):
                        v[i][0] = (v[i][0] - center_ind[0]) / center_ind[0]
                        v[i][2] = (v[i][2] - center_ind[0]) / center_ind[0]
                        v[i][1] = (v[i][1] - center_ind[1]) / center_ind[1]
                        v[i][3] = (v[i][3] - center_ind[1]) / center_ind[1]
            elif k == 'K':
                camKs = self._K2feature(data[k])
                v = []
                for i, (start, end) in enumerate(data['split']):
                    v.append(camKs[i:i+1, :].expand(end-start, -1))
                v = torch.cat(v, 0)
            elif k in ['analytic_code', 'structured_implicit_vector', 'blob_center']:
                if k == 'analytic_code':
                    v = data['structured_implicit'].analytic_code
                elif k == 'structured_implicit_vector':
                    v = data['structured_implicit'].vector
                elif k == 'blob_center':
                    # get world_sys points from blob centers
                    bdb3D_form = self._get_bdb3D_form(data)
                    centers = data['structured_implicit'].all_centers.clone()
                    centers[:, :, 2] *= -1
                    v = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                v = v.reshape([v.shape[0], -1])
            elif k == 'ldif_phy':
                assert type == 'rel'
                bdb3D_form = self._get_bdb3D_form(data)
                structured_implicit = data['structured_implicit']
                ldif_center, ldif_coef = data['obj_center'], data['obj_coef']

                # get world_sys points from blob centers
                centers = data['structured_implicit'].all_centers.clone()
                centers[:, :, 2] *= -1
                obj_samples = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                element_count = centers.shape[1]

                # put points to other objects' coor
                max_sample_points = (data['split'][:, 1] - data['split'][:, 0]).max() * element_count
                other_obj_samples = torch.zeros([len(obj_samples), max_sample_points, 3], device=centers.device)
                for start, end in data['split']:
                    other_obj_sample = obj_samples[start:end].reshape(1, -1, 3).expand(end - start, -1, -1)
                    other_obj_samples[start:end, :other_obj_sample.shape[1]] = other_obj_sample
                other_obj_samples = recover_points_to_obj_sys(bdb3D_form, other_obj_samples, ldif_center, ldif_coef)
                other_obj_samples[:, :, 2] *= -1

                # get sdf of blob centers from other objects
                est_sdf = data['mgn'](
                    samples=other_obj_samples,
                    structured_implicit=structured_implicit.dict(),
                    apply_class_transfer=False,
                )['global_decisions'] + 0.07

                # reshape into relation features
                v = [est_sdf[start:end, :(end - start) * element_count].reshape(-1, element_count)
                     for start, end in data['split']]
                v = torch.cat(v)

            else:
                raise NotImplementedError

            if type == 'obj' or k in ('g_features', 'ldif_phy'):
                features.append(v)
            else:
                features_rel = []
                for start, end in data['split']:
                    features_rel.append(torch.stack(
                        [torch.cat([loc1, loc2], -1)
                         for loc1 in v[start:end]
                         for loc2 in v[start:end]]))
                features.append(torch.cat(features_rel, 0))
        return torch.cat(features, -1)

    def _get_layout_features(self, data):
        features = []
        keys = self.lo_features
        for k in keys:
            if k in ['pitch_reg_result', 'roll_reg_result', 'pitch_cls_result',
                     'roll_cls_result', 'lo_ori_reg_result', 'lo_ori_cls_result',
                     'lo_centroid_result', 'lo_coeffs_result', 'lo_afeatures']:
                v = data[k]
            elif k == 'K':
                v = self._K2feature(data[k])
            else:
                raise NotImplementedError

            features.append(v)
        return torch.cat(features, -1)

    def _get_map(self, data):
        device = data['g_features'].device
        split = data['split']
        obj_num = split[-1][-1] + split.shape[0]  # number of objects and layouts
        obj_obj_map = torch.zeros([obj_num, obj_num])  # mapping of obj/lo vertices with connections
        rel_inds = []  # indexes of vertices connected by relation nodes
        rel_masks = []  # mask of relation features for obj/lo vertices connected by relation nodes
        obj_masks = torch.zeros(obj_num, dtype=torch.bool)  # mask of object vertices
        lo_masks = torch.zeros(obj_num, dtype=torch.bool)  # mask of layout vertices
        for lo_index, (start, end) in enumerate(split):
            start = start + lo_index  # each subgraph has Ni object vertices and 1 layout vertex
            end = end + lo_index + 1  # consider layout vertex, Ni + 1 vertices in total
            obj_obj_map[start:end, start:end] = 1  # each subgraph is a complete graph with self circle
            obj_ind = torch.arange(start, end, dtype=torch.long)
            subj_ind_i, obj_ind_i = torch.meshgrid(obj_ind, obj_ind)  # indexes of each vertex in the subgraph
            rel_ind_i = torch.stack([subj_ind_i.reshape(-1), obj_ind_i.reshape(-1)], -1)
            rel_mask_i = rel_ind_i[:, 0] != rel_ind_i[:, 1]  # vertices connected by relation nodes should be different
            rel_inds.append(rel_ind_i[rel_mask_i])
            rel_masks.append(rel_mask_i)
            obj_masks[start:end - 1] = True  # for each subgraph, first Ni vertices are objects
            lo_masks[end - 1] = True  # for each subgraph, last 1 vertex is layout

        rel_inds = torch.cat(rel_inds, 0)
        rel_masks = torch.cat(rel_masks, 0)

        subj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])  # [sum(Ni + 1), sum((Ni + 1) ** 2)]
        obj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])
        # map from subject (an object or layout vertex) to predicate (a relation vertex)
        subj_pred_map.scatter_(0, (rel_inds[:, 0].view(1, -1)), 1)
        # map from object (an object or layout vertex) to predicate (a relation vertex)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].view(1, -1)), 1)

        return rel_masks.to(device), obj_masks.to(device), lo_masks.to(device), \
               obj_obj_map.to(device), subj_pred_map.to(device), obj_pred_map.to(device)

    def forward(self, output):

        bdb2D_pos_resize = output['bdb2D_pos_resize']
        split = output['split']
        size_feat, ori_feat, centroid_feat, offset_feat = output['odn_fc_features']   # 128
        cam_feat, lo_feat = output['lo_fc_features']   # 1024
        res_feat = output['lo_res_features']

        rel_feat, lo_mask_cls_score = self.rel(res_feat, bdb2D_pos_resize, split)

        # branch for camera parameters
        cam = self.fc_1(cam_feat)
        cam = self.relu_1(cam)
        cam = self.dropout_1(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0: self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN: self.PITCH_BIN * 2]
        roll_reg = cam[:, self.PITCH_BIN * 2: self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN: self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

        # branch for layout orientation, centroid and coefficients
        lo = self.fc_layout(torch.cat((cam_feat, lo_feat), dim=1))
        lo_fused = self.relu_1(lo)
        lo = self.dropout_1(lo_fused)
        # branch for layout orientation
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu_1(lo_ori)
        lo_ori = self.dropout_1(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]

        # branch for layout centroid and coefficients
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu_1(lo_ct)
        lo_ct = self.dropout_1(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]

        # branch to predict the size
        size = self.fc1(torch.cat((size_feat, rel_feat), dim=1))
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)

        # branch to predict the orientation
        ori = self.fc3(torch.cat((ori_feat, rel_feat), dim=1))
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]

        # branch to predict the centroid
        centroid = self.fc5(torch.cat((centroid_feat, rel_feat), dim=1))
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]

        # branch to predict the 2D offset
        offset = self.fc_off_1(torch.cat((offset_feat, rel_feat), dim=1))
        offset = self.relu_1(offset)
        offset = self.dropout_1(offset)
        offset = self.fc_off_2(offset)


        if self.res_output:
            size += output['size_reg_result']
            ori_reg += output['ori_reg_result']
            ori_cls += output['ori_cls_result']
            centroid_reg += output['centroid_reg_result']
            centroid_cls += output['centroid_cls_result']
            offset += output['offset_2D_result']

            pitch_reg += output['pitch_reg_result']
            pitch_cls += output['pitch_cls_result']
            roll_reg += output['roll_reg_result']
            roll_cls += output['roll_cls_result']
            lo_ori_reg += output['lo_ori_reg_result']
            lo_ori_cls += output['lo_ori_cls_result']
            lo_centroid += output['lo_centroid_result']
            lo_coeffs += output['lo_coeffs_result']

        return {'size_reg_result': size, 'ori_reg_result': ori_reg,
                'ori_cls_result': ori_cls, 'centroid_reg_result': centroid_reg,
                'centroid_cls_result': centroid_cls, 'offset_2D_result': offset,
                'pitch_reg_result': pitch_reg, 'pitch_cls_result': pitch_cls, 'roll_reg_result': roll_reg,
                'roll_cls_result': roll_cls, 'lo_ori_reg_result': lo_ori_reg, 'lo_ori_cls_result': lo_ori_cls,
                'lo_centroid_result': lo_centroid, 'lo_coeffs_result': lo_coeffs, 'lo_mask_cls_score': lo_mask_cls_score}
