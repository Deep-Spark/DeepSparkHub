import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
import random

dist_th = 8e-3# threshold from HardNet, negative descriptor pairs with the distances lower than this threshold are treated as false negatives
eps_l2_norm = 1e-10
eps_sqrt = 1e-6

def cal_l2_distance_matrix(x, y, flag_sqrt=True):
    ''''distance matrix of x with respect to y, d_ij is the distance between x_i and y_j'''
    D = torch.abs(2 * (1 - torch.mm(x, y.t())))
    if flag_sqrt:
        D = torch.sqrt(D + eps_sqrt)
    return D

def read_UBC_patch_opencv(train_root, sz_patch):
    patch = []
    file = sorted(os.listdir(train_root))
    nb_file = len(file)
    sz_patch_raw = 64
    flag_resize = False
    if sz_patch_raw != sz_patch:
        flag_resize = True
    for i, img_file in enumerate(file):
        if img_file.find('bmp') > -1:
            img = cv2.imread(os.path.join(train_root, img_file), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape#height is row, width is column
            print('reading:{} of {}'.format(i, nb_file), end='\r')
            for v in range(0, img_height, sz_patch_raw):  # Vertival
                for h in range(0, img_width, sz_patch_raw):  # Horizontal
                    patch_temp = img[v:v+sz_patch_raw, h:h+sz_patch_raw]#64x64
                    if flag_resize:
                        patch_temp = cv2.resize(patch_temp, (sz_patch, sz_patch))
                    patch.append(patch_temp)
    print('reading patch:{} of {}'.format(i, nb_file))


    return np.expand_dims(np.array(patch), 1)# N*1*sz_patch*sz_patch

def read_UBC_pointID(train_root):
    pointID = []
    with open(os.path.join(train_root, 'info.txt')) as f:
        for line in f:
            id = int(line.split(' ')[0])
            pointID.append(id)
            print('reading pointID:id{}'.format(id), end='\r')
    print('max ID:{}'.format(id))
    return np.array(pointID)

def read_hpatches_patch_opencv(train_root, sz_patch):
    sz_patch_raw = 65
    if sz_patch_raw != sz_patch:
        flag_resize = True
    patch = []
    scene_file = sorted(os.listdir(train_root))
    num_scene = len(scene_file)
    sets = ['e', 'h', 't']
    for i, scene in enumerate(scene_file):#
        print('reading:{} of {} scene'.format(i, num_scene), end='\r')
        img_all = []
        img_all.append(cv2.imread(os.path.join(train_root, scene, 'ref.png'), cv2.IMREAD_GRAYSCALE))
        for j, set in enumerate(sets):
            for k in range(1, 6):
                img_all.append(cv2.imread(os.path.join(train_root, scene, set + str(k)+'.png'), cv2.IMREAD_GRAYSCALE))

        img_height, _ = img_all[0].shape  # height is row, width is column

        for v in range(0, img_height, sz_patch_raw): # only Vertival
            for i, img in enumerate(img_all):
                patch_temp = img[v:v+sz_patch_raw]
                if flag_resize:
                    patch_temp = cv2.resize(patch_temp, (sz_patch, sz_patch))
                patch.append(patch_temp)

    return np.expand_dims(np.array(patch), 1)# N*1*sz_patch*sz_patch

def cal_index_train_all(index_unique_label, inb_label_each_batch, epoch_max):
    index_train = []
    num_label = len(index_unique_label)
    num_patch = 0
    for i in range(num_label):
        num_patch += index_unique_label[i].size

    nb_batch_each_epoch = int(np.ceil(num_label/inb_label_each_batch))

    for e_loop in range(epoch_max):
        #loop over each epoch
        each_epoch_index = []
        print('calculating train index:epoch {} of {}'.format(e_loop+1, epoch_max), end='\r')
        for b_loop in range(nb_batch_each_epoch):
            each_batch_index = []
            #loop over each batch in each epoch
            for i in range(inb_label_each_batch):
                j_max = len(index_unique_label[i])
                for j in range(j_max):
                    each_batch_index.append(index_unique_label[i][j])
            each_epoch_index.append(each_batch_index)
            index_unique_label = np.roll(index_unique_label, -inb_label_each_batch)
        index_train.append(each_epoch_index)
        np.random.shuffle(index_unique_label)
    return np.array(index_train)

def cal_index_train(index_unique_label, num_label_each_batch, num_img_each_label, epoch_max):
    print('calculating index_train...')
    #ensure input is numpy array
    index_train = []

    num_label = len(index_unique_label)
    num_patch = 0
    for i in range(num_label):
        num_patch += index_unique_label[i].size

    index_index = [i for i in range(num_label)]#for random shuffule

    index_unique_label0 = index_unique_label.copy()

    sz_batch = num_img_each_label*num_label_each_batch
    num_batch_each_epoch = int(num_patch/sz_batch)
    for e_loop in range(epoch_max):
        #loop over each epoch
        each_epoch_index = []
        print('calculating train index:epoch {} of {}'.format(e_loop,epoch_max))
        for b_loop in tqdm(range(num_batch_each_epoch)):#num_batch_each_epoch
            #loop over each batch in each epoch
            each_batch_index = []
            for i in range(num_label_each_batch):
                #loop over each label in each batch
                if len(index_unique_label[i]) < num_img_each_label:
                    np.random.shuffle(index_unique_label0[i])
                    index_unique_label[i] = index_unique_label0[i]
                    #refill the variable if less than num_img_each_label
                for j in range(num_img_each_label):
                    each_batch_index.append(index_unique_label[i][0])
                    if b_loop + i + j == 0:
                        unique_label_temp = np.delete(index_unique_label[i], [0])
                        index_unique_label = list(index_unique_label)
                        index_unique_label[i] = unique_label_temp
                        index_unique_label = np.array(index_unique_label, dtype=object)
                    else:
                        index_unique_label[i] = np.delete(index_unique_label[i], [0])

            each_epoch_index.append(each_batch_index)
            index_unique_label = np.roll(index_unique_label, -num_label_each_batch)
            index_unique_label0 = np.roll(index_unique_label0, -num_label_each_batch)

            if (b_loop+1) % int(np.ceil(num_label/num_label_each_batch)) == 0:
                random.shuffle(index_index)
                index_unique_label = index_unique_label[index_index]
                index_unique_label0 = index_unique_label0[index_index]

        index_train.append(each_epoch_index)

    return np.array(index_train)

def load_UBC_for_train(data_root, train_set, sz_patch=32, nb_pt_each_batch=512, nb_pat_per_pt=2, epoch_max=200, flag_load_index=True): # all outputs are numpy arrays
    train_root = os.path.join(data_root, train_set)
    file_data_train = os.path.join(train_root, train_set + '_sz' + str(sz_patch) + '.npz')
    file_index_train = os.path.join(train_root, train_set + '_index_train_ID' + str(nb_pt_each_batch) + '_pat' + str(nb_pat_per_pt) + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = data['patch']
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        patch = read_UBC_patch_opencv(train_root, sz_patch)
        pointID = read_UBC_pointID(train_root)
        index_unique_ID = []  # it is a list
        pointID_unique = np.unique(pointID)
        for id in pointID_unique:
            index_unique_ID.append(np.argwhere(pointID == id).squeeze())
        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=np.array(index_unique_ID, dtype=object))
    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            if nb_pat_per_pt == -1:
                index_train = cal_index_train_all(index_unique_ID, nb_pt_each_batch, epoch_max)
            else:
                index_train = cal_index_train(index_unique_ID, nb_pt_each_batch, nb_pat_per_pt, epoch_max)
            np.save(file_index_train, index_train)

    return torch.from_numpy(patch), pointID, index_train

def extract_100K_test(patch_train,pointID_train,test_root):
    patch_loc = []
    index_test = []
    with open(os.path.join(test_root, 'm50_100000_100000_0.txt')) as f:
        for line in f:
            id = line.split(' ')
            patch_loc.append(int(id[0]))
            patch_loc.append(int(id[3]))
            index_test.append([int(id[0]), int(id[3])])

    patch_loc = np.array(patch_loc)
    patch_loc = np.unique(patch_loc)
    pointID_test = pointID_train[patch_loc]
    patch_test = patch_train[patch_loc]
    for i in range(len(index_test)):
        index_test[i][0] = np.argwhere(patch_loc == index_test[i][0]).squeeze()
        index_test[i][1] = np.argwhere(patch_loc == index_test[i][1]).squeeze()

    return patch_test, pointID_test, np.array(index_test)

def load_UBC_for_test(data_root, test_set, sz_patch=32): # all outputs are numpy arrays
    test_root = os.path.join(data_root, test_set)
    file_data_test = os.path.join(test_root, test_set + '_sz' + str(sz_patch) + '_100k_test.npz')

    if os.path.exists(file_data_test):
        print('Test data of {} already exists!'.format(test_set))
        data = np.load(file_data_test, allow_pickle=True)
        patch_test = data['patch']
        pointID_test = data['pointID']
        index_test = data['index']#Only tesy data have attribuate 'index'
    else:
        file_data_train = os.path.join(test_root, test_set + '_sz' + str(sz_patch) + '.npz')
        if os.path.exists(file_data_train):
            # If there is train data
            data_train = np.load(file_data_train, allow_pickle=True)
            patch_train = data_train['patch']
            pointID_train = data_train['pointID']
            del data_train
        else:
            # First generate the train data
            print(test_set)
            patch_train = read_UBC_patch_opencv(test_root, sz_patch)
            pointID_train = read_UBC_pointID(test_root)
            np.savez(file_data_train, patch=patch_train, pointID=pointID_train)

        patch_test, pointID_test, index_test = extract_100K_test(patch_train, pointID_train, test_root)
        np.savez(file_data_test, patch=patch_test, pointID=pointID_test, index=index_test)

    return patch_test, pointID_test, index_test

def load_hpatches_for_train(data_root, sz_patch, nb_pt_each_batch, nb_pat_each_pt, epoch_max, flag_load_index=True):
    train_set = 'hpatches'
    train_root = os.path.join(data_root, 'hpatches-benchmark-master/data/hpatches-release')
    save_root = os.path.join(data_root, 'hpatches-benchmark-master/data/')

    file_data_train = os.path.join(save_root, train_set + '_sz' + str(sz_patch) + '.npz')
    file_index_train = os.path.join(save_root, train_set + '_index_train_ID' + str(nb_pt_each_batch) + '_pat' + str(nb_pat_each_pt) + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = data['patch']
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        patch = read_hpatches_patch_opencv(train_root, sz_patch)
        num_uniqueID = int(len(patch)/16)
        pointID_unique = np.array(range(0, num_uniqueID))
        pointID = []
        for i in range(0, num_uniqueID):
            for j in range(0, 16):
                pointID.append(i)

        pointID = np.array(pointID)
        index_unique_ID = []  # it is a list
        for id in pointID_unique:
            index_unique_ID.append(np.argwhere(pointID == id).squeeze())
        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=index_unique_ID)

    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            index_train = cal_index_train(index_unique_ID, nb_pt_each_batch, nb_pat_each_pt, epoch_max)

            np.save(file_index_train, index_train)

    return torch.from_numpy(patch), pointID, index_train

def load_hpatches_split_train(data_root, sz_patch, nb_pt_each_batch=512, nb_pat_each_pt=2, epoch_max=100, split_name='a', flag_load_index=True, flag_std_filter=True):
    train_set = 'hpatches'
    train_root = os.path.join(data_root, 'hpatches-benchmark-master/data/hpatches-release')
    save_root = os.path.join(data_root, 'hpatches-benchmark-master/data/')
    file_data_train = os.path.join(save_root, train_set + '_sz' + str(sz_patch) + '_split_' + split_name + '.npz')
    file_index_train = os.path.join(save_root, train_set + '_index_train_ID' + str(nb_pt_each_batch) + '_pat' + str(nb_pat_each_pt) + '_split_' + split_name + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = torch.from_numpy(data['patch'])
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        split_file = os.path.join(data_root, 'hpatches-benchmark-master/tasks/splits/splits.json')
        with open(split_file) as f:
            split = json.load(f)
        train_split = split[split_name]['train']
        patch = read_hpatches_patch_split_opencv(train_root, train_split, sz_patch)
        num_uniqueID = int(len(patch)/16)
        pointID_unique = np.array(range(0, num_uniqueID))
        pointID = []
        for i in range(0, num_uniqueID):
            for j in range(0, 16):
                pointID.append(i)
        pointID = np.array(pointID)

        if flag_std_filter:
            patch_raw = read_hpatches_patch_split_opencv(train_root, train_split, sz_patch=65)
            patch_std = compute_patch_contrast(patch_raw)  # return numpy array
            indice_high_std = np.argwhere(patch_std > 0).squeeze()
            patch = patch[indice_high_std]
            pointID = pointID[indice_high_std]
            pointID_unique = np.unique(pointID)


        index_unique_ID = []  # it is a list
        for id in pointID_unique:
            indice_ID = np.argwhere(pointID == id).squeeze()
            if len(indice_ID) >= 2:
                index_unique_ID.append(indice_ID)

        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=index_unique_ID)

    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            index_train = cal_index_train(index_unique_ID, nb_pt_each_batch, nb_pat_each_pt, epoch_max)
            np.save(file_index_train, index_train)

    return patch, pointID, index_train

def read_hpatches_patch_split_opencv(train_root, scene_train, sz_patch):
    sz_patch_raw = 65
    flag_resize = False
    if sz_patch_raw != sz_patch:
        flag_resize = True
    patch = []
    num_scene = len(scene_train)

    sets = ['e', 'h', 't']
    for i, scene in enumerate(scene_train):
        print('reading:{} of {} scene'.format(i, num_scene), end='\r')
        img_all = []
        img_all.append(cv2.imread(os.path.join(train_root, scene, 'ref.png'), cv2.IMREAD_GRAYSCALE))
        for j, set in enumerate(sets):
            for k in range(1, 6):
                img_all.append(cv2.imread(os.path.join(train_root, scene, set + str(k)+'.png'), cv2.IMREAD_GRAYSCALE))

        img_height, _ = img_all[0].shape  # height is row, width is column

        for v in range(0, img_height, sz_patch_raw): # only Vertival
            for i, img in enumerate(img_all):
                patch_temp = img[v:v+sz_patch_raw]
                if flag_resize:
                    patch_temp = cv2.resize(patch_temp, (sz_patch, sz_patch))
                patch.append(patch_temp)

    return np.expand_dims(np.array(patch), 1)# N*1*sz_patch*sz_patch

def data_aug(patch, num_ID_per_batch):
    # sz = patch.size()
    patch.squeeze_()
    patch = patch.numpy()
    for i in range(0, num_ID_per_batch):
        if random.random() > 0.5:
            nb_rot = np.random.randint(1, 4)
            patch[2*i] = np.rot90(patch[2*i], nb_rot)
            patch[2*i+1] = np.rot90(patch[2*i + 1], nb_rot)


        if random.random() > 0.5:
            patch[2 * i] = np.flipud(patch[2 * i])
            patch[2 * i + 1] = np.flipud(patch[2 * i + 1])

        # if random.random() > 0.5:
        #     patch[2 * i] = np.fliplr(patch[2*i])
        #     patch[2 * i + 1] = np.fliplr(patch[2*i + 1])



    patch = torch.from_numpy(patch)
    patch.unsqueeze_(1)
    return patch

def cal_fpr95(desc,pointID,pair_index):
    dist = desc[pair_index[:, 0],:] - desc[pair_index[:, 1],:]
    dist.pow_(2)
    dist = torch.sqrt(torch.sum(dist,1))
    pairSim = pointID[pair_index[:, 0]] - pointID[pair_index[:, 1]]
    pairSim = torch.Tensor(pairSim)
    dist_pos = dist[pairSim == 0]
    dist_neg = dist[pairSim != 0]
    dist_pos, indice = torch.sort(dist_pos)
    loc_thr = int(np.ceil(dist_pos.numel() * 0.95))
    thr = dist_pos[loc_thr]
    fpr95 = float(dist_neg.le(thr).sum())/dist_neg.numel()
    return fpr95
