# import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,re
from tqdm import tqdm
import hdf5storage as hdf5
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from module import rgnn
from utils import *
# 数据划分
def sub_split(data_dict, label_dict, sub_test, sub_train=None):
    if sub_test not in range(1, 16):
        raise ValueError('sub for test must be in range(1, 16)')
    if sub_train is not None:
        if all(1 <= x <= 16 for x in sub_train) == False:
            raise ValueError('sub for train must be in range(1, 16)')
        if sub_test in sub_train:
            raise ValueError('sub for test and train can not overlap')

    x_train = []
    y_train = []
    x_test = None
    y_test = None
    train_data_list = []
    test_data_list = []
    train_sub_list = []  # 训练集的被试列表
    for i in range(1, 16):
        for key in data_dict.keys():
            if key.startswith('__'): continue
            sub_num = int(key.split('_')[1])
            if sub_num != i: continue
            # print('shape:', key, data_dict[key].shape, label_dict[key].shape)
            if sub_num != sub_test:
                if sub_train is not None:
                    if sub_num in sub_train:
                        x_train.append(data_dict[key])
                        y_train.append(label_dict[key])
                        train_sub_list.append(key)
                    else:
                        pass
                else:
                    x_train.append(data_dict[key])
                    y_train.append(label_dict[key])
                    train_sub_list.append(key)
            else:
                print('{} for test sub'.format(key))
                x_test = data_dict[key]
                y_test = label_dict[key]
    print('train sub list:', train_sub_list)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    # M = 3394*3=10182
    x_train = x_train.reshape(-1, x_train.shape[-2], x_train.shape[-1])  # M*62*5
    y_train = y_train.reshape(-1, y_train.shape[-1])  # N*3
    x_test = x_test.reshape(-1, x_test.shape[-2], x_test.shape[-1])  # M*62*5
    y_test = y_test.reshape(-1, y_test.shape[-1])  # N*3
    print(f'x_train:{x_train.shape},  y_train:{y_train.shape},  x_test:{x_test.shape},  y_test:{y_test.shape}')

    for i in range(len(x_train)):
        train_data_list.append(
            Data(x=torch.from_numpy(x_train[i]).float(), edge_index=edge_index, edge_attr=edge_weight,
                 y=torch.Tensor(y_train[i])))
    for i in range(len(x_test)):
        test_data_list.append(Data(x=torch.from_numpy(x_test[i]).float(), edge_index=edge_index, edge_attr=edge_weight,
                                   y=torch.Tensor(y_test[i])))

    assert (len(train_data_list) == x_train.shape[0] and len(train_data_list) == y_train.shape[0])
    assert (len(test_data_list) == x_test.shape[0] and len(test_data_list) == y_test.shape[0])

    return train_data_list, test_data_list

# 评估
def evaluation(model,data_loader,device):
    total, correct = 0, 0
    for data in data_loader:
        x = data.x.to(device)
        index = data.edge_index.to(device)
        batch = data.batch.to(device)
        y = data.y.to(device) # (bs*3,)或者(bs*1,)
        y_pre, _ = model(x, index, batch)
        total += len(torch.unique(batch))
        if LABEL_DL:
            y = y.reshape(y_pre.shape)
            for i in range(y.size(0)):
                if torch.abs(y[i][0]-y_pre[i][0])>0.25:
                    continue
                if torch.abs(y[i][1]-y_pre[i][1])>0.25:
                    continue
                if torch.abs(y[i][2]-y_pre[i][2])>0.25:
                    continue
                correct = correct+1
        else:
            _, y_pre = torch.max(y_pre, dim=1)
            correct += (y_pre == y).sum().item()
    # print('correct:', correct, 'total:', total)
    return 100 * correct / total
    # return correct

# 训练
def train(model, data_loader, domain_attr, criterion, optimizer, device):
    running_loss = 0.0
    domain_loss = 0.0
    for data in data_loader:
        batch_int = data.batch.size(0)
        if domain_attr==0:
            train_domain = torch.zeros((batch_int)).long().to(device) # 训练集的域标签为0
        if domain_attr==1:
            train_domain = torch.ones((batch_int)).long().to(device) # 测试集的域标签为1
        x = data.x.to(device)    
        index = data.edge_index.to(device)
        batch = data.batch.to(device)
        y = data.y.to(device) # (bs*3,)
        # print(y)
        optimizer.zero_grad()
        y_pre, domain = model(x, index, batch)
        if domain is not None:
            loss1 = criterion(domain, train_domain)
        else:
            loss1 = 0

        if LABEL_DL:
            loss2 = F.kl_div(y_pre.softmax(dim=-1).log(), y.reshape(y_pre.shape).softmax(dim=-1), reduction='sum')
        else:
            y = y.long()
            loss2 = criterion(y_pre, y)
        loss = loss1 + loss2
        domain_loss += loss1
        running_loss += loss
        loss.backward()
        optimizer.step()
    # print('loss:', running_loss, 'domain_loss:', domain_loss)
    return running_loss, domain_loss


# seed_path = 'seed_processed'
# data_path = os.path.join(seed_path, 'dic_all_subjects.mat')
# label_path = os.path.join(seed_path, 'dic_all_subjects_label.mat')
root_path = 'data\seed_all'
data_path = os.path.join(root_path, 'data_dic.mat')
label_path = os.path.join(root_path, 'label_dic.mat')
data_dic = hdf5.loadmat(data_path)
label_dic = hdf5.loadmat(label_path)

LABEL_DL = True
for key in data_dic.keys():
    if key.startswith('__'):
        continue
    data_dic[key] = extend_normal(data_dic[key]) # 归一化; 2025*62*25
    if LABEL_DL:
        label_dic[key] = np.array([EmotionDL(label) for label in label_dic[key][0]]) # 1*2025 -> 2025*3
    else:
        label_dic[key] = label_dic[key].T
        # pass
    print(key, data_dic[key].shape, label_dic[key].shape)


ch_names = ['Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6'
            ,'F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5'
            ,'C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2'
            ,'CP4','CP6','TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7'
            ,'PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']
global_pairs = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F5', 'F6'], ['FC5', 'FC6'], ['C5','C6'],['CP5', 'CP6'],['P5', 'P6'],['PO5', 'PO6'],['O1', 'O2']]
global_connections = get_ASM_electrode_indices(ch_names, global_pairs).tolist() # 全局连接
pos_arr = np.load('location_62.npy')
# 生成邻接矩阵
num_node = 62
A = get_adj(num_node, pos_arr, global_connections)
edge_index, edge_weight = get_edge_info(A)

result_path = 'result'
info_file_path = os.path.join(result_path, "info.txt")
fig_file_path = os.path.join(result_path, "fig")
if not os.path.exists(result_path):
    os.makedirs(result_path)
    os.makedirs(fig_file_path)

seed_all(seed = 123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 150
learning_rate = 5e-4
dropout = 0.2
num_hidden = 30
num_class = 3
# loss smoothing
LS = 0.05
DA = 1

with open(info_file_path, 'a') as f:
    f.write('\n-----------------------------------------------------------\n')
    f.write('RGNN with seed dataset\n')
    f.write(f'batch_size: {batch_size}, epochs: {epochs}, learning_rate: {learning_rate}, dropout: {dropout}, num_hidden: {num_hidden}\n')
    f.write(f'num_class: {num_class}\n')
    if DA:
        f.write('domain adaptation is applied\n')
    if LS:
        f.write(f'label smoothing rate: {LS}\n')
    f.write('-----------------------------------------------------------\n')

def main(sub = 1):
    print('-----------------------------------------------------------')
    # sub=1  # 1-15

    sub_list = [i for i in range(1, 16) if i != sub]
    train_data_list, test_data_list = sub_split(data_dic, label_dic, sub, sub_list)
    # print(len(train_data_list), len(test_data_list))

    
    train_loader = DataLoader(train_data_list, batch_size=batch_size)
    test_loader = DataLoader(test_data_list, batch_size=batch_size)
    for batch_data in train_loader:
        print('batch data in trainer:')
        print(batch_data)
        break
    for batch_data in test_loader:
        print('batch data in tester:')
        print(batch_data)
        break
    input_dim = batch_data.x.shape[1]
    print(f'input dim: {input_dim} ,Now training start...')
    print('-----------------------------------------------------------')

    model = rgnn(num_in=input_dim, num_hidden=num_hidden, K=2, num_class=num_class, dropout=dropout, domain_adaptation=DA,
                 weight=edge_weight)

    if LS != 0:
        loss = LabelSmoothing(LS)
        # loss = CE_Label_Smooth_Loss(LS)
    else:
        loss = nn.CrossEntropyLoss()
    
    # opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_acc = 0
    losses = []
    train_accuracies = []
    domain_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        train_loss, domain_loss1 = train(model, train_loader, 0, loss, opt, device)
        test_loss, domain_loss2 = train(model, test_loader, 1, loss, opt, device)
        loss_total = ((train_loss + test_loss)/(len(train_loader)+len(test_loader))).item()
        domain_loss = ((domain_loss1 + domain_loss2)/(len(train_loader)+len(test_loader)))
        
        train_acc = evaluation(model, train_loader, device)
        test_acc = evaluation(model, test_loader, device)

        print(f'[{epoch + 1:03d}|{epochs:03d}], Loss: {loss_total:.4f}, Domain loss: {domain_loss:.4f}, Train accuracy: {train_acc:.2f}%, Test accuracy: {test_acc:.2f}%')
        losses.append(loss_total)
        domain_losses.append(domain_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"best acc [{best_acc:.2f}%] in epoch {epoch + 1}")
    
    print(f"Final best test accuracy for sub_{sub} tester: {best_acc:.2f}%")
    with open(info_file_path, 'a') as f:
        f.write(f"sub_{sub}: {best_acc:.2f}%\n")
    print(f'the result of sub{sub} has been saved in info.txt')
    print('-----------------------------------------------------------\n')

    epochX = range(1, epochs + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochX, losses, '-r', label='Training loss')
    # plt.plot(epochX, domain_losses, '-', label='Domain loss')
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', color='gray', alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochX, domain_losses, '-g', label='Domain loss')
    plt.title('domain loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True, linestyle='--', color='gray', alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochX, train_accuracies, '-', label='Training accuracy')
    plt.plot(epochX, test_accuracies, '-', label='Test accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', color='gray', alpha=0.3)
    plt.legend()

    plt.tight_layout()
    # plt.savefig('RGNN_seed_epoch_info.png')
    plt.savefig(os.path.join(fig_file_path, f"sub_{sub}_epoch_info.png"))

if __name__ == '__main__':
    for sub in range(1, 16):
    # sub = 4 # 1-15
        main(sub)