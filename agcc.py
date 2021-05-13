from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
import opt
import scipy.sparse as sp
from collections import Counter


# torch.cuda.set_device(1)

class AttentionLayer(nn.Module):
    def __init__(self, last_dim, n_num):
        super(AttentionLayer, self).__init__()
        self.n_num = n_num
        self.fc1 = nn.Linear(n_num * last_dim, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, n_num)
        self.attention = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.T = 10
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        attention_sample = self.attention(x / self.T)
        attention_view = torch.mean(attention_sample, dim=0, keepdim=True).squeeze()
        return attention_view

class FusionLayer(nn.Module):
    def __init__(self, last_dim, n_num=2):
        super(FusionLayer, self).__init__()
        self.n_num = n_num
        self.attentionLayer = AttentionLayer(last_dim, n_num)
    def forward(self, x, k):
        y = torch.cat((x, k), 1)
        weights = self.attentionLayer(y)
        x_TMP = weights[0] * x + weights[1] * k
        return x_TMP


def dot_product(z):
    if opt.args.name == "usps" or opt.args.name == "hhar" or opt.args.name == "reut":
        adj1 = F.softmax(F.relu(torch.mm(z, z.transpose(0, 1))), dim=1)
    else:
        adj1 = torch.sigmoid(torch.mm(z, z.transpose(0, 1)))
        adj1 = adj1.add(torch.eye(adj1.shape[0]).to(device))
        adj1 = normalize(adj1)
    return adj1

def normalize(mx):

    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h1, dec_h2, dec_h3



class DTFU(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(DTFU, self).__init__()

        # autoencoder for intra information
        self.ael = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.ael.load_state_dict(torch.load(opt.args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_z)
        #self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        self.gnn_6 = GNNLayer(n_clusters, n_dec_1)
        self.gnn_7 = GNNLayer(n_dec_1, n_dec_2)
        self.gnn_8 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_9 = GNNLayer(n_dec_3, n_input)

        self.fuse1 = FusionLayer(n_enc_1)
        self.fuse2 = FusionLayer(n_enc_2)
        self.fuse3 = FusionLayer(n_enc_3)
        self.fuse4 = FusionLayer(n_z)

        self.fuse5 = FusionLayer(n_dec_1)
        self.fuse6 = FusionLayer(n_dec_2)
        self.fuse7 = FusionLayer(n_dec_3)


        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, dec_1, dec_2, dec_3 = self.ael(x)

        sigma = 0.5

        #GCN Module
        h = self.gnn_1(x, adj)

        h = self.fuse1(h, tra1)
        adj1 = dot_product(h)
        h = self.gnn_2(h, ((1 - sigma) * adj1 + sigma * adj))

        # h = self.fuse2(h, tra2)
        # adj1 = dot_product(tra2)
        # h = self.gnn_3(h, ((1 - sigma) * adj1 + sigma * adj))
        #
        # adj1 = dot_product(h)
        # h = self.fuse3(h, tra3)
        # h = self.gnn_4(h, ((1 - sigma) * adj1 + sigma * adj))

        h = self.fuse4(h, z)
        adj1 = dot_product(h)
        h1 = self.gnn_5(h, ((1 - sigma) * adj1 + sigma * adj), active=False)


        predict = F.softmax(h1, dim=1)


        h = self.gnn_6(h1, ((1 - sigma) * adj1 + sigma * adj))
        h = self.fuse5(h, dec_1)
        h = self.gnn_7(h, ((1 - sigma) * adj1 + sigma * adj))
        h = self.fuse6(h, dec_2)
        h = self.gnn_9(h, ((1 - sigma) * adj1 + sigma * adj))

        A_pred = dot_product(h)



        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, A_pred


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_dtfu(dataset):
    model = DTFU(500, 500, 2000, 2000, 500, 500,
                 n_input=opt.args.n_input,
                 n_z=opt.args.n_z,
                 n_clusters=opt.args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    # KNN Graph
    adj = load_graph(opt.args.name, opt.args.k)
    adj = adj.to(device)

    data = torch.Tensor(dataset.x).to(device)

    # cluster parameter initiate

    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z, _, _, _ = model.ael(data)

    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    M = np.zeros((700, 4))
    for epoch in range(700):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            M[epoch, 0], M[epoch, 1], M[epoch, 2], M[epoch, 3] = eva(y, res2, str(epoch) + 'Z')
            # eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _, A_pred = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        re_graphloss = F.mse_loss(A_pred, adj.to_dense())


        loss = opt.args.lambda_v1 * kl_loss + opt.args.lambda_v2 * ce_loss + re_loss + opt.args.lambda_v3 * re_graphloss
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('acc:', np.max(M[:, 0]))
    print('nmi:', np.max(M[:, 1]))
    print('ari:', np.max(M[:, 2]))
    print('f1:', np.max(M[:, 3]))


if __name__ == "__main__":

    opt.args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(opt.args.cuda))
    device = torch.device("cuda" if opt.args.cuda else "cpu")


    opt.args.pretrain_path = 'data/{}.pkl'.format(opt.args.name)
    dataset = load_data(opt.args.name)

    if opt.args.name == 'usps':
        opt.args.lr = 1e-4
        opt.args.n_clusters = 10
        opt.args.n_input = 256
        opt.args.seed = 5
        setup_seed(opt.args.seed_usps)


    if opt.args.name == 'hhar':
        opt.args.k = 5
        opt.args.n_clusters = 6
        opt.args.n_input = 561
        setup_seed(opt.args.seed_hhar)


    if opt.args.name == 'reut':
        opt.args.k = 1
        opt.args.lr = 1e-4
        opt.args.n_clusters = 4
        opt.args.n_input = 2000
        setup_seed(opt.args.seed_reut)
        opt.args.lambda_v1 = 1.0
        opt.args.lambda_v2 = 0.01
        opt.args.lambda_v3 = 0.1

    if opt.args.name == 'acm':
        opt.args.k = None
        opt.args.n_clusters = 3
        opt.args.n_input = 1870
        setup_seed(opt.args.seed_acm)



    if opt.args.name == 'dblp':
        opt.args.lr = 1e-3
        opt.args.k = None
        opt.args.n_clusters = 4
        opt.args.n_input = 334
        setup_seed(opt.args.seed_dblp)

    if opt.args.name == 'cite':
        opt.args.lr = 5e-5
        opt.args.k = None
        opt.args.n_clusters = 6
        opt.args.n_input = 3703
        opt.args.seed = 4
        setup_seed(opt.args.seed_cite)


    print(opt.args)
    train_dtfu(dataset)
