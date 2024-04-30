# -*- coding: utf-8 -*-

# UPDATE:
# @Time   : 2022/9/16
# @Author : Jiabo Zhuang
# @Email  : 1283266308@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole_gnn.model.abstract_recommender import SocialRecommender
from recbole_gnn.model.layers import LightGCNConv
from recbole_gnn.model.layers import BipartiteGCNConv
import torch.nn.functional as F
from recbole.model.layers import MyattLayer
class SocialGCN(SocialRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SocialGCN, self).__init__(config, dataset)

        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']
        self.maskrat=config['maskrat']
        self.alpha = config['alpha']
        self.beta = config['beta']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']
        self.attdim = 1

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.bipartite_gcn_conv = BipartiteGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data and net data
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)
        self.net_edge_index, self.net_edge_weight = dataset.get_norm_net_adj_mat(row_norm=False)
        self.net_edge_index, self.net_edge_weight = self.net_edge_index.to(self.device), self.net_edge_weight.to(self.device)
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.attlayer1=MyattLayer(self.latent_dim *2,self.attdim)
        self.attlayer2=MyattLayer(self.latent_dim *2,self.attdim)
        self.attlayer3=MyattLayer(self.latent_dim ,self.attdim)


    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, perturbed=0,aug=0):
        all_embeddings = self.get_ego_embeddings()
        '''if aug == 1:
            random_noise = torch.rand_like(all_embeddings, device=all_embeddings.device)
            all_embeddings = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
        if aug == 2:
            random_noise = torch.rand_like(all_embeddings, device=all_embeddings.device)
            all_embeddings = all_embeddings - torch.sign(all_embeddings) * F.normalize(random_noise, dim=-1) * self.eps'''
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            #user_embeddings_pre,item_embeddings_pre=torch.split(all_embeddings,[self.n_users, self.n_items])
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)  # user-item图卷积
            user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
            #user_embeddings1 = user_embeddings.clone()
            #in_embeddings1 = torch.cat((user_embeddings_pre,user_embeddings1),dim=1)
            user_embeddings2 = self.bipartite_gcn_conv((user_embeddings, user_embeddings),self.net_edge_index.flip([0]), self.net_edge_weight,size=(self.n_users, self.n_users))# user-user图卷积
            #in_embeddings2 = torch.cat((user_embeddings_pre,user_embeddings2),dim=1)
            '''if layer_idx ==0: #第一层的注意力机制
                in_embeddings = torch.stack((in_embeddings1,in_embeddings2),dim=1)
                attsignal = self.attlayer1(in_embeddings).unsqueeze(dim=2)
                ori_embeddings = torch.stack((user_embeddings1,user_embeddings2),dim=1)
                attuser = torch.mul(attsignal,ori_embeddings)    # [batch_size, num_graphs, emb_dim]
                all_embeddings[:self.n_users] = torch.sum(attuser, dim=1)  # [batch_size, emb_dim]
            if layer_idx ==1:#第二层的注意力机制 如果需要第三层要额外写，这里只写两层
                in_embeddings = torch.stack((in_embeddings1,in_embeddings2),dim=1)
                attsignal = self.attlayer2(in_embeddings).unsqueeze(dim=2)
                ori_embeddings = torch.stack((user_embeddings1,user_embeddings2),dim=1)
                attuser = torch.mul(attsignal,ori_embeddings)    # [batch_size, num_graphs, emb_dim]
                all_embeddings[:self.n_users] = torch.sum(attuser, dim=1)  # [batch_size, emb_dim]'''
            all_embeddings[:self.n_users] = (all_embeddings[:self.n_users] * self.beta + user_embeddings2 * self.alpha)

            if perturbed == 1:
                random_noise1 = torch.rand_like(all_embeddings, device=all_embeddings.device)
                all_embeddings = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise1, dim=-1) * self.eps
                #random_mask1 = torch.rand_like(all_embeddings,device=all_embeddings.device)
                #random_mask1 = random_mask1.ge(self.maskrat)
                #all_embeddings = all_embeddings.masked_fill(random_mask1,0)
            if perturbed == 2:
                random_noise2 = torch.rand_like(all_embeddings, device=all_embeddings.device)
                all_embeddings = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise2,dim=-1) * self.eps
                #random_mask2= torch.rand_like(all_embeddings, device=all_embeddings.device)
                #random_mask2 = random_mask2.ge(self.maskrat)
                #all_embeddings = all_embeddings.masked_fill(random_mask2, 0)'''
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        # lightgcn_all_embeddings = torch.cat((embeddings_list[0],embeddings_list[1]),1)


        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=1,aug=0)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=2,aug=0)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])
        #loss = mf_loss + self.reg_weight * reg_loss
        loss = mf_loss + self.reg_weight * reg_loss+ self.cl_rate * (user_cl_loss + item_cl_loss)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
