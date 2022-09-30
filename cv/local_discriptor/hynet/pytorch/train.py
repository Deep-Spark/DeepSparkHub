import ast
import argparse
from utils import *
from model import HyNet
import torch.optim as optim

class Loss_HyNet():

    def __init__(self, device, num_pt_per_batch, dim_desc, margin, alpha, is_sosr, knn_sos=8):
        self.device = device
        self.margin = margin
        self.alpha = alpha
        self.is_sosr = is_sosr
        self.num_pt_per_batch = num_pt_per_batch
        self.dim_desc = dim_desc
        self.knn_sos = knn_sos
        self.index_desc = torch.LongTensor(range(0, num_pt_per_batch))
        self.index_dim = torch.LongTensor(range(0, dim_desc))
        diagnal = torch.eye(num_pt_per_batch)
        self.mask_pos_pair = diagnal.eq(1).float().to(self.device)
        self.mask_neg_pair = diagnal.eq(0).float().to(self.device)

    def sort_distance(self):
        L = self.L.clone().detach()
        L = L + 2 * self.mask_pos_pair
        L = L + 2 * L.le(dist_th).float()

        R = self.R.clone().detach()
        R = R + 2 * self.mask_pos_pair
        R = R + 2 * R.le(dist_th).float()

        LR = self.LR.clone().detach()
        LR = LR + 2 * self.mask_pos_pair
        LR = LR + 2 * LR.le(dist_th).float()

        self.indice_L = torch.argsort(L, dim=1)
        self.indice_R = torch.argsort(R, dim=0)
        self.indice_LR = torch.argsort(LR, dim=1)
        self.indice_RL = torch.argsort(LR, dim=0)
        return

    def triplet_loss_hybrid(self):
        L = self.L
        R = self.R
        LR = self.LR
        indice_L = self.indice_L[:, 0]
        indice_R = self.indice_R[0, :]
        indice_LR = self.indice_LR[:, 0]
        indice_RL = self.indice_RL[0, :]
        index_desc = self.index_desc

        dist_pos = LR[self.mask_pos_pair.bool()]
        dist_neg_LL = L[index_desc, indice_L]
        dist_neg_RR = R[indice_R, index_desc]
        dist_neg_LR = LR[index_desc, indice_LR]
        dist_neg_RL = LR[indice_RL, index_desc]
        dist_neg = torch.cat((dist_neg_LL.unsqueeze(0),
                              dist_neg_RR.unsqueeze(0),
                              dist_neg_LR.unsqueeze(0),
                              dist_neg_RL.unsqueeze(0)), dim=0)
        dist_neg_hard, index_neg_hard = torch.sort(dist_neg, dim=0)
        dist_neg_hard = dist_neg_hard[0, :]
        # scipy.io.savemat('dist.mat', dict(dist_pos=dist_pos.cpu().detach().numpy(), dist_neg=dist_neg_hard.cpu().detach().numpy()))

        loss_triplet = torch.clamp(self.margin + (dist_pos + dist_pos.pow(2)/2*self.alpha) - (dist_neg_hard + dist_neg_hard.pow(2)/2*self.alpha), min=0.0)

        self.num_triplet_display = loss_triplet.gt(0).sum()

        self.loss = self.loss + loss_triplet.sum()
        self.dist_pos_display = dist_pos.detach().mean()
        self.dist_neg_display = dist_neg_hard.detach().mean()

        return

    def norm_loss_pos(self):
        diff_norm = self.norm_L - self.norm_R
        self.loss += diff_norm.pow(2).sum().mul(0.1)

    def sos_loss(self):
        L = self.L
        R = self.R
        knn = self.knn_sos
        indice_L = self.indice_L[:, 0:knn]
        indice_R = self.indice_R[0:knn, :]
        indice_LR = self.indice_LR[:, 0:knn]
        indice_RL = self.indice_RL[0:knn, :]
        index_desc = self.index_desc
        num_pt_per_batch = self.num_pt_per_batch
        index_row = index_desc.unsqueeze(1).expand(-1, knn)
        index_col = index_desc.unsqueeze(0).expand(knn, -1)

        A_L = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_R = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_LR = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)

        A_L[index_row, indice_L] = 1
        A_R[indice_R, index_col] = 1
        A_LR[index_row, indice_LR] = 1
        A_LR[indice_RL, index_col] = 1

        A_L = A_L + A_L.t()
        A_L = A_L.gt(0).float()
        A_R = A_R + A_R.t()
        A_R = A_R.gt(0).float()
        A_LR = A_LR + A_LR.t()
        A_LR = A_LR.gt(0).float()
        A = A_L + A_R + A_LR
        A = A.gt(0).float() * self.mask_neg_pair

        sturcture_dif = (L - R) * A
        self.loss = self.loss + sturcture_dif.pow(2).sum(dim=1).add(eps_sqrt).sqrt().sum()

        return

    def compute(self, desc_L, desc_R, desc_raw_L, desc_raw_R):
        self.desc_L = desc_L
        self.desc_R = desc_R
        self.desc_raw_L = desc_raw_L
        self.desc_raw_R = desc_raw_R
        self.norm_L = self.desc_raw_L.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.norm_R = self.desc_raw_R.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.L = cal_l2_distance_matrix(desc_L, desc_L)
        self.R = cal_l2_distance_matrix(desc_R, desc_R)
        self.LR = cal_l2_distance_matrix(desc_L, desc_R)

        self.loss = torch.Tensor([0]).to(self.device)

        self.sort_distance()
        self.triplet_loss_hybrid()
        self.norm_loss_pos()
        if self.is_sosr:
            self.sos_loss()

        return self.loss, self.dist_pos_display, self.dist_neg_display

class Loss_SOSNet():

    def __init__(self, device, num_pt_per_batch, dim_desc, margin, knn_sos=8):
        self.device = device
        self.margin = margin
        self.num_pt_per_batch = num_pt_per_batch
        self.dim_desc = dim_desc
        self.knn_sos = knn_sos
        self.index_desc = torch.LongTensor(range(0, num_pt_per_batch))
        self.index_dim = torch.LongTensor(range(0, dim_desc))
        diagnal = torch.eye(num_pt_per_batch)
        self.mask_pos_pair = diagnal.eq(1).float().to(self.device)
        self.mask_neg_pair = diagnal.eq(0).float().to(self.device)

    def sort_distance(self):
        L = self.L.clone().detach()
        L = L + 2 * self.mask_pos_pair
        L = L + 2 * L.le(dist_th).float()

        R = self.R.clone().detach()
        R = R + 2 * self.mask_pos_pair
        R = R + 2 * R.le(dist_th).float()

        LR = self.LR.clone().detach()
        LR = LR + 2 * self.mask_pos_pair
        LR = LR + 2 * LR.le(dist_th).float()

        self.indice_L = torch.argsort(L, dim=1)
        self.indice_R = torch.argsort(R, dim=0)
        self.indice_LR = torch.argsort(LR, dim=1)
        self.indice_RL = torch.argsort(LR, dim=0)
        return

    def triplet_loss(self):
        L = self.L
        R = self.R
        LR = self.LR
        indice_L = self.indice_L[:, 0]
        indice_R = self.indice_R[0, :]
        indice_LR = self.indice_LR[:, 0]
        indice_RL = self.indice_RL[0, :]
        index_desc = self.index_desc

        dist_neg_hard_L = torch.min(LR[index_desc, indice_LR], L[index_desc, indice_L])
        dist_neg_hard_R = torch.min(LR[indice_RL, index_desc], R[indice_R, index_desc])
        dist_neg_hard = torch.min(dist_neg_hard_L, dist_neg_hard_R)
        dist_pos = LR[self.mask_pos_pair.bool()]
        loss = torch.clamp(self.margin + dist_pos - dist_neg_hard, min=0.0)

        loss = loss.pow(2)

        self.loss = self.loss + loss.sum()
        self.dist_pos_display = dist_pos.detach().mean()
        self.dist_neg_display = dist_neg_hard.detach().mean()

        return

    def sos_loss(self):
        L = self.L
        R = self.R
        knn = self.knn_sos
        indice_L = self.indice_L[:, 0:knn]
        indice_R = self.indice_R[0:knn, :]
        indice_LR = self.indice_LR[:, 0:knn]
        indice_RL = self.indice_RL[0:knn, :]
        index_desc = self.index_desc
        num_pt_per_batch = self.num_pt_per_batch
        index_row = index_desc.unsqueeze(1).expand(-1, knn)
        index_col = index_desc.unsqueeze(0).expand(knn, -1)

        A_L = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_R = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_LR = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)

        A_L[index_row, indice_L] = 1
        A_R[indice_R, index_col] = 1
        A_LR[index_row, indice_LR] = 1
        A_LR[indice_RL, index_col] = 1

        A_L = A_L + A_L.t()
        A_L = A_L.gt(0).float()
        A_R = A_R + A_R.t()
        A_R = A_R.gt(0).float()
        A_LR = A_LR + A_LR.t()
        A_LR = A_LR.gt(0).float()
        A = A_L + A_R + A_LR
        A = A.gt(0).float() * self.mask_neg_pair

        sturcture_dif = (L - R) * A
        self.loss = self.loss + sturcture_dif.pow(2).sum(dim=1).add(eps_sqrt).sqrt().sum()

        return

    def compute(self, desc_l, desc_r):
        self.loss = torch.Tensor([0]).to(self.device)
        self.L = cal_l2_distance_matrix(desc_l, desc_l)
        self.R = cal_l2_distance_matrix(desc_r, desc_r)
        self.LR = cal_l2_distance_matrix(desc_l, desc_r)
        self.sort_distance()
        self.triplet_loss()
        self.sos_loss()

        return self.loss, self.dist_pos_display, self.dist_neg_display

def train_net(desc_name, nb_batch_per_epoch):
    net.train()
    running_loss = 0.0
    running_dist_pos = 0.0
    running_dist_neg = 0.0
    for batch_loop in range(nb_batch_per_epoch):

        index_batch = index_train[epoch_loop][batch_loop]
        batch = patch_train[index_batch]
        batch = batch.to(torch.float32)
        if flag_dataAug:
            batch = data_aug(batch, num_pt_per_batch)

        batch = batch.to(device)
        desc_L, desc_raw_L = net(batch[0::2], mode='train')
        desc_R, desc_raw_R = net(batch[1::2], mode='train')
        if desc_name == 'HyNet':
            loss, dist_pos, dist_neg = loss_desc.compute(desc_L, desc_R, desc_raw_L, desc_raw_R)
        elif desc_name == 'SOSNet' or desc_name == 'HardNet':
            loss, dist_pos, dist_neg = loss_desc.compute(desc_L, desc_R)

        running_loss = running_loss + loss.item()
        running_dist_pos += dist_pos.item()
        running_dist_neg += dist_neg.item()
        if (batch_loop+1) % 100 ==0 or batch_loop + 1 == nb_batch_per_epoch:
            print('epoch {}: {}/{}: dist_pos: {:.4f}, dist_neg: {:.4f}, loss: {:.4f}'.format(
                epoch_loop + 1,
                batch_loop + 1,
                nb_batch_per_epoch,
                running_dist_pos / (batch_loop + 1),
                running_dist_neg / (batch_loop + 1),
                running_loss / (batch_loop + 1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return

def test_net(device, net, patch, pointID, index, dim_desc=128, sz_batch=500):
    net.eval()
    nb_patch = pointID.size
    nb_loop = int(np.ceil(nb_patch/sz_batch))
    desc = torch.zeros(nb_patch, dim_desc)
    with torch.set_grad_enabled(False):
        for i in range(nb_loop):
            st = i * sz_batch
            en = np.min([(i + 1) * sz_batch, nb_patch])
            batch = patch[st:en].to(device)
            out_desc = net(batch, mode='eval')
            out_desc = out_desc.to('cpu')
            desc[st:en] = out_desc
            # print(': {} of {}'.format(i, nb_loop), end='\r')


    fpr95 = cal_fpr95(desc, pointID, index)
    return fpr95

parser = argparse.ArgumentParser(description='pyTorch descNet')
parser.add_argument('--data_root', type=str, default='data')# path containing the UBC and HPatches data set

parser.add_argument('--train_set', type=str, default='liberty')# notredame, liberty, yosemite, hpatches,
parser.add_argument('--train_split', type=str, default='a')# full

parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--sz_patch', type=int, default=32)
parser.add_argument('--num_pt_per_batch', type=int, default=512)
parser.add_argument('--dim_desc', type=int, default=128)
parser.add_argument('--nb_pat_per_pt', type=int, default=2)
parser.add_argument('--epoch_max', type=int, default=200)
parser.add_argument('--margin', type=float, default=1.2)

parser.add_argument('--flag_dataAug', type=ast.literal_eval, default=True)
parser.add_argument('--is_sosr', type=ast.literal_eval, default=False)
parser.add_argument('--knn_sos', type=int, default=8)

parser.add_argument('--optim_method', type=str, default='Adam')
parser.add_argument('--lr_scheduler', type=str, default='None')#CosineAnnealing

parser.add_argument('--desc_name', type=str, default='HyNet')
parser.add_argument('--alpha', type=float, default=2)

parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--drop_rate', type=float, default=0.3)

args = parser.parse_args()

data_root = args.data_root
train_set = args.train_set
sz_patch = args.sz_patch
epoch_max = args.epoch_max
num_pt_per_batch = args.num_pt_per_batch
nb_pat_per_pt = args.nb_pat_per_pt
num_pt_per_batch = args.num_pt_per_batch
dim_desc = args.dim_desc
margin = args.margin
drop_rate = args.drop_rate
is_sosr = args.is_sosr
knn_sos = args.knn_sos
flag_dataAug = args.flag_dataAug
optim_method = args.optim_method
lr_scheduler = args.lr_scheduler
alpha = args.alpha
desc_name = args.desc_name
train_split = args.train_split
lr = args.lr

# get save folder name
folder_name = desc_name + '_' + train_set

if train_set == 'hpatches':
    folder_name += '_split_' + train_split

folder_name += '_sz_' + str(sz_patch)
folder_name += '_pt_' + str(num_pt_per_batch)
folder_name += '_pat_' + str(nb_pat_per_pt)
folder_name += '_dim_' + str(dim_desc)

if args.is_sosr or args.desc_name == 'SOSNet':
    folder_name += '_SOSR' + '_KNN_' + str(knn_sos)

if args.desc_name == 'HyNet':
    folder_name += '_alpha_' + str(alpha)

folder_name += '_margin_' + str(margin)
folder_name += '_drop_' + str(drop_rate)
folder_name += '_lr_' + str(lr)
folder_name += '_' + optim_method + '_' + lr_scheduler

if flag_dataAug:
    folder_name += '_aug'

if len(args.suffix) > 0:# for debugging
    folder_name += '-' + args.suffix

net_dir = os.path.join('model', folder_name)
print(net_dir)

if not os.path.exists(net_dir):
    os.makedirs(net_dir)
else:
    print('path already exists')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# data preparation
# For each train batch, we first sample #num_pt_per_batch 3D points, and then for each of the 3D point we sample #nb_pat_per_pt patches
ubc_subset = ['yosemite', 'notredame', 'liberty']
if train_set == 'liberty' or args.train_set == 'notredame' or args.train_set == 'yosemite':
    patch_train, pointID_train, index_train = load_UBC_for_train(data_root, train_set,
                                                                 sz_patch,
                                                                 num_pt_per_batch, nb_pat_per_pt,
                                                                 epoch_max)
    test_set = []
    for val in ubc_subset:
        if val != train_set:
            test_set.append(val)
elif train_set == 'hpatches':
    if args.train_split == 'all':
        patch_train, pointID_train, index_train = load_hpatches_for_train(args.data_root,
                                                                          args.sz_patch,
                                                                          args.num_pt_per_batch,
                                                                          args.nb_pat_per_pt,
                                                                          args.epoch_max)
    else:
        patch_train, pointID_train, index_train = load_hpatches_split_train(data_root,
                                                                            sz_patch,
                                                                            num_pt_per_batch,
                                                                            nb_pat_per_pt,
                                                                            epoch_max,
                                                                            split_name=train_split)
    test_set = ['yosemite', 'notredame']
nb_batch_per_epoch = len(index_train[0])# Each epoch has equal number of batches

patch_test = {}
pointID_test = {}
index_test = {}
for i, val in enumerate(test_set):
    patch_test[val], pointID_test[val], index_test[val] = load_UBC_for_test(args.data_root, val, args.sz_patch)
    patch_test[val] = torch.from_numpy(patch_test[val])
    patch_test[val] = patch_test[val].to(torch.float32)
    index_test[val] = index_test[val]

# model, optimizer
net = HyNet(dim_desc=dim_desc, drop_rate=drop_rate)
net.to(device)

if optim_method == 'Adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
elif optim_method == 'SGD':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9)

if lr_scheduler == 'CosineAnnealing':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=epoch_max,
                                                           eta_min=1e-7,
                                                           last_epoch=-1)

# file names
file_fpr95 = 'fpr95_'
for i, val in enumerate(test_set):
    file_fpr95 = file_fpr95 + val + '_'

file_fpr95_best = file_fpr95[0:-1] + '_best.npy'
file_fpr95 = file_fpr95[0:-1] + '.npy'
file_fpr95 = os.path.join(net_dir, file_fpr95)
file_fpr95_best = os.path.join(net_dir, file_fpr95_best)
net_best_name = os.path.join(net_dir, 'net-best.pth')

# descriptor type
if desc_name == 'HyNet':
    loss_desc = Loss_HyNet(device, num_pt_per_batch, dim_desc, margin, alpha, is_sosr, knn_sos)
elif desc_name == 'SOSNet':
    loss_desc = Loss_SOSNet(device, num_pt_per_batch, dim_desc, margin, knn_sos)

# start training
fpr95 = []
for epoch_loop in range(args.epoch_max):
    #train
    train_net(desc_name, nb_batch_per_epoch)

    if lr_scheduler != 'None':
        scheduler.step()

    net_name = os.path.join(net_dir, 'net-epoch-{}.pth'.format(epoch_loop + 1))
    torch.save(net.state_dict(), net_name)
    # validation
    fpr95_per_epoch = []
    for i, val in enumerate(test_set):
        fpr_temp = test_net(device, net, patch_test[val], pointID_test[val], index_test[val], args.dim_desc)
        print('{}: {}'.format(val, fpr_temp))
        fpr95_per_epoch.append(fpr_temp)
    if len(fpr95_per_epoch) > 0:
        fpr95.append(fpr95_per_epoch)
        np.save(file_fpr95, fpr95)
        fpr_avg = np.mean(np.array(fpr95_per_epoch))
        if epoch_loop == 0:
            fpr_avg_best = fpr_avg
            epoch_best = 0
        else:
            if fpr_avg_best > fpr_avg:
                fpr_avg_best = fpr_avg
                fpr_best = fpr95_per_epoch.copy()
                fpr_best.append(epoch_loop+1)
                torch.save(net.state_dict(), net_best_name)
                np.save(file_fpr95_best, fpr_best)
