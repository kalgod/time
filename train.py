import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from itertools import groupby

import logging
import sys

# 设置 logging 配置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("temp1.txt",mode='w'),  # 将日志输出到 log.txt 文件
        # logging.StreamHandler(sys.stdout)  # 也输出到控制台
    ]
)

def predict_next_sequence(x):
    # 确保输入的序列长度为5
    if len(x) != 5:
        raise ValueError("Input sequence must be of length 5.")
    
    # 检查 x 是否为递增或递减
    is_increasing = all(x[i] < x[i + 1] for i in range(len(x) - 1))
    is_decreasing = all(x[i] > x[i + 1] for i in range(len(x) - 1))
    
    # 线性插值函数
    def linear_interpolation(start, end, n):
        return [start + i * (end - start) / (n + 1) for i in range(1, n + 1)]
    
    if is_increasing:
        # 完全递增
        future_values = linear_interpolation(x[-1], x[-1] + (x[-1] - x[-2]), 5)
    elif is_decreasing:
        # 完全递减
        future_values = linear_interpolation(x[-1], x[-1] - (x[-2] - x[-1]), 5)
    else:
        # 根据最后两个数的递增和递减关系
        if x[-1] > x[-2]:
            # 如果最后一个数大于倒数第二个数，进行线性插值
            future_values = linear_interpolation(x[-1], x[-1] + (x[-1] - x[-2]), 5)
        else:
            # 如果最后一个数小于倒数第二个数，进行线性插值
            future_values = linear_interpolation(x[-1], x[-1] - (x[-2] - x[-1]), 5)
    
    return future_values

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets, inputs):
        # inputs: 形状为 [batch_size, in_seq]
        # predictions: 模型的预测输出 [batch_size, out_seq]
        # targets: 实际的目标输出 [batch_size, out_seq]

        # 检查输入 x 的最后两个元素
        last_two = inputs[:, -2:]  # 取最后两个元素 [batch_size, 2]
        
        # 计算条件
        is_increasing = (last_two[:, 1] > last_two[:, 0]).float()  # [batch_size]
        
        # 对于每个样本，检查预测值是否递增或递减
        # 先计算预测值的递增/递减性质
        is_prediction_increasing = (predictions[:, 1:] > predictions[:, :-1]).float()  # [batch_size, out_seq - 1]
        
        # 如果输入是递增的，没有必要惩罚递增
        increasing_loss = torch.mean((is_prediction_increasing - 1) ** 2, dim=1)  # 惩罚递增的损失

        # 如果输入是递减的，没有必要惩罚递减
        is_prediction_decreasing = (predictions[:, 1:] < predictions[:, :-1]).float()  # [batch_size, out_seq - 1]

        decreasing_loss = torch.mean(is_prediction_decreasing ** 2, dim=1)  # 惩罚递减的损失
        
        # 组合损失
        total_loss = torch.where(is_increasing > 0, increasing_loss, decreasing_loss).mean()

        return total_loss

def remove_adjacent_duplicates(arr, col_index=1):
    if arr.size == 0:  # 处理空数组
        return arr
    
    # 初始化结果列表，包含第一行
    result = [arr[0]]  
    
    for i in range(1, arr.shape[0]):  # 从第二行开始遍历
        if arr[i, col_index] != arr[i - 1, col_index]:  # 仅当当前行的指定列与前一行不同
            result.append(arr[i])  # 添加到结果列表
    
    return np.array(result)  # 将结果转换回NumPy数组

def gaussian_variant(x, a, b, mu, sigma):
    """高斯分布变体模型"""
    return a + b * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def adjust_pred(x_data):
    x_coords = np.arange(len(x_data))
    # 估计初始参数
    mu = len(x_data)/2  # 均值
    mu = estimate_mu(np.arange(len(x_data)), x_data)
    b_ini=np.max(x_data)-np.min(x_data)
    sigma = np.std(x_data/b_ini)  # 标准差
    initial_guess = [np.min(x_data) , b_ini, mu, sigma*10]  # a, b, mu, sigma 的初值
    print(f"initial_guess: {initial_guess}")

    # 拟合高斯分布变体
    bounds = ([0, -5,-5,0], [1, 5,10,10])  # 参数的取值范围
    params, _ = curve_fit(gaussian_variant, x_coords, x_data, p0=initial_guess, bounds=bounds)

    # 提取拟合的参数
    a, b,mu, sigma = params
    print(f"拟合得到的参数: a={a}, b={b}, mu={mu}, sigma={sigma}")

    # 生成未来的结果
    future_n = args.out_len  # 预测未来 5 个值
    future_coords = np.arange(len(x_data), len(x_data) + future_n)

    # 使用拟合后的高斯函数预测未来的值
    future_values = gaussian_variant(future_coords, *params)
    pred=np.array(future_values)
    return pred

def estimate_mu(x, y):
    # return x[-1]
    diff = np.diff(y)  # 计算增减速度
    turning_points = np.where(np.abs(diff) < 1e-3)[0]  # 找到速度接近零的位置
    if len(turning_points) > 0:
        return x[turning_points[0]]  # 返回第一个增长速度接近零的位置
    else:
        # 如果没有找到速度接近零的位置，根据趋势判断mu
        if diff[-1]*diff[-2] > 0:  # 如果最后速度仍在增长，数据在分布左侧
            return x[-1]
        else:  # 如果最后速度在减小，数据在分布右侧
            return (x[-1] + x[0]) / 2  # 估计mu在数据范围之外

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def PLCC(pred, true):
    res=[]
    for i in range (len(pred)):
        a=pred[i].squeeze()
        b=true[i].squeeze()
        tmp=pearsonr(a, b)[0]
        if (np.isnan(tmp)): continue
        res.append(tmp)
        # print(i,a,b,res[-1],"\n")
    return np.mean(res)

def SRCC(pred, true):
    res=[]
    for i in range (len(pred)):
        a=pred[i].squeeze()
        b=true[i].squeeze()
        tmp=spearmanr(a, b)[0]
        if (np.isnan(tmp)): continue
        res.append(tmp)
    return np.mean(res)

def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / (true + 1e-8)))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) /( true + 1e-8)))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    # corr = CORR(pred, true)
    plcc = PLCC(pred, true)
    srcc = SRCC(pred, true)

    return mae, mse, rmse, mape, mspe,plcc,srcc


scaler = StandardScaler()

def discret_data(all_data, discret):
    for i in discret:
        renumbered = {}
        tmp=all_data[:,i]
        unique_values = np.unique(tmp)
        for idx, value in enumerate(sorted(unique_values)):
            # print(idx,value)
            renumbered[value] = idx
        tmp=np.vectorize(renumbered.get)(tmp)
        all_data[:,i]=tmp
    return all_data

def clean(file_name,out_name):
    df = pd.read_csv(file_name)
    all_data = df.to_numpy()
    print(all_data.shape,all_data[:2])
    all_data=all_data[all_data[:,7]!=-1]
    all_data=all_data[all_data[:,-3]!=0]
    mask = ~np.any(pd.isna(all_data), axis=1)
    all_data = all_data[mask]
    # all_data=all_data[:100000]
    discret=[3,4,5,8,10,11,12,13,14,15,17,21,22]
    all_data=discret_data(all_data, discret)
    print(all_data.shape,all_data[:2])

    timestamps = all_data[:, 29]
    timestamps_in_seconds = timestamps / 1000.0
    dates = pd.to_datetime(timestamps_in_seconds, unit='s')
    all_data[:, 29] = dates.strftime('%Y/%m/%d %H:%M:%S')

    df.columns.values[[3,29]] = df.columns.values[[29,3]]
    df.columns.values[[-3,-2]] = df.columns.values[[-2,-3]]
    all_data[:,[3,29]]=all_data[:,[29,3]]
    all_data[:,[-3,-2]]=all_data[:,[-2,-3]]
    all_data[:,[-1,-2]]=all_data[:,[-2,-1]]

    columns_to_remove = [0, 1,2,16]
    all_data = np.delete(all_data, columns_to_remove, axis=1)
    tmp_col=df.columns.tolist()
    tmp_col=np.delete(tmp_col, columns_to_remove)

    tmp_col[0]="date"
    tmp_col[-1]="OT"

    # all_data=all_data[:,[0,-1]]
    # tmp_col=tmp_col[[0,-1]]

    print(all_data.shape,tmp_col.shape,tmp_col)
    df = pd.DataFrame(all_data, columns=tmp_col)
    df.to_csv(out_name, index=False)
    return all_data,tmp_col

def generate(args,file_name):
    df = pd.read_csv(file_name)
    all_data = df.to_numpy()
    all_data=remove_adjacent_duplicates(all_data)
    ori_data=all_data
    
    # print(all_data.shape,all_data[:2])
    res=[]
    for i in range (len(all_data)-args.in_len-args.out_len+1):
        num=all_data[i][-2]
        num_after=all_data[i+args.in_len+args.out_len-1][-2]
        # print(i,num,num_after)
        if num_after!=num: continue
        res.append(i)
    res=np.array(res)
    all_data=all_data[:,1].reshape(-1,1)
    all_data=scaler.fit_transform(all_data)
    all_data=np.array(all_data,dtype=np.float32)
    return ori_data,all_data,res

def split(args,file_name):
    ori_data,all_data,all_idx=generate(args,file_name)
    # np.random.shuffle(all_idx)
    train_len=int(len(all_idx)*0.7)
    train_idx = all_idx[:train_len]
    test_idx = all_idx[train_len:]
    print(len(all_idx),len(train_idx),len(test_idx))
    all_data=torch.from_numpy(all_data).float()
    return ori_data,all_data,train_idx,test_idx

class CustomDataset(Dataset):
    def __init__(self, all_data,all_idx,args):
        self.all_data = all_data
        self.all_idx = all_idx
        self.args=args

    def __len__(self):
        return len(self.all_idx)

    def __getitem__(self, idx):
        idx=self.all_idx[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x=self.all_data[idx:idx+self.args.in_len]
        y=self.all_data[idx+self.args.in_len:idx+self.args.in_len+self.args.out_len]
        if (args.fea_len==1):
            x=x[:,-1]
        y=y[:,-1]
        return x,y

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x)  # [batch_size, seq_len, input_dim]
        K = self.key(x)    # [batch_size, seq_len, input_dim]
        V = self.value(x)  # [batch_size, seq_len, input_dim]

        # 计算注意力分数
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attn_scores = attn_scores / (self.input_dim ** 0.5)  # 归一化
        attn_weights = self.softmax(attn_scores)  # [batch_size, seq_len, seq_len]

        # 计算加权和
        output = torch.bmm(attn_weights, V)  # [batch_size, seq_len, input_dim]
        return output

class DNNModel(nn.Module):
    def __init__(self, in_seq,n, out_seq):
        super(DNNModel, self).__init__()
        self.in_seq = in_seq
        self.self_attention = SelfAttention(in_seq)  # 输入维度也是 in_seq

        self.fc1 = nn.Linear(in_seq, 64)  # 第一层全连接层
        self.fc2 = nn.Linear(64, 32)      # 第二层全连接层
        self.fc3 = nn.Linear(32, out_seq)  # 输出层

    def forward(self, x):
        # x: [batch_size, in_seq]，需要扩展到 [batch_size, in_seq, in_seq]
        x = x.unsqueeze(1)  # [batch_size, 1, in_seq] 
        x = x.expand(-1, self.in_seq, -1)  # [batch_size, in_seq, in_seq] 

        attention_output = self.self_attention(x)  # [batch_size, in_seq, in_seq]

        # 对注意力输出在序列维度上求平均
        attention_output = attention_output.mean(dim=1)  # [batch_size, in_seq]

        # 将注意力输出送入全连接层
        x = F.relu(self.fc1(attention_output))  # 使用 ReLU 激活函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def inverse(data):
    res=[]
    for row in data:
        row=scaler.inverse_transform(row.reshape(-1,1))
        res.append(row)
    return np.array(res)

def calculate_plcc(y_true, y_pred):
    all_plcc=[]
    for i in range (len(y_true)):
        a=y_pred[i].squeeze()
        b=y_true[i].squeeze()
        """计算 Pearson 相关系数"""
        a_mean = torch.mean(a)
        b_mean = torch.mean(b)

        # 计算协方差
        covariance = torch.mean((a - a_mean) * (b - b_mean))
        
        # 计算标准差
        std_true = torch.std(a)
        std_pred = torch.std(b)
        
        # 计算 PLCC
        plcc = covariance / (std_true * std_pred+1e-9)
        all_plcc.append(plcc)
    plcc=torch.mean(torch.stack(all_plcc))
    return plcc

def test(args,test_loader,model):
    pred_list = []
    true_list = []
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(args.device)
            outputs = model(batch_x)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            pred_ori=inverse(pred.numpy())
            true_ori=inverse(true.numpy())

            pred_list.append(pred_ori)
            true_list.append(true_ori)
            # print(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")
            # print(pred.shape,true.shape,pred_ori.shape,true_ori.shape)
            # exit(0)
    pred_list=np.concatenate(pred_list, axis=0)
    true_list=np.concatenate(true_list, axis=0)
    mae, mse, rmse, mape, mspe,plcc,srcc=metric(pred_list, true_list)
    return mae, mse, rmse, mape, mspe,plcc,srcc

def plot(a,b,name):
    plt.figure(figsize=(10, 5))
    seconds=np.arange(len(a))
    plt.plot(seconds, a, label="Ground Truth Score", color='green')
    plt.plot(seconds,b, label="Prediction", color='blue', linestyle='--')
    # plt.plot(seconds, score2, label=f"VASNet:loss{loss2:.4f}", color='red', linestyle='--')
    # plt.plot(seconds, score3, label=f"SL_module:loss{loss3:.4f}", color='gray', linestyle='--')
    # plcc, plcc_p_value = pearsonr(score, gtscore)
    # srcc, srcc_p_value = spearmanr(score, gtscore)
    # # plcc_idx, plcc_idx_p_value = pearsonr(idx_final_scores, idx_gt_scores)
    # # srcc_idx, srcc_idx_p_value = spearmanr(idx_final_scores, idx_gt_scores)
    # print(f"{name}: (PLCC): {plcc}, p-value: {plcc_p_value}, (SRCC): {srcc}, p-value: {srcc_p_value}")

    plt.title(f'{name}: Score Comparison')
    plt.xlabel('Second')
    plt.ylabel('Score')
    plt.legend()

    img_path = os.path.join("./result", f"{name}_comparison.png")
    plt.savefig(img_path)
    plt.close()

def test_video(args,all_data,model):
    pred_list = []
    true_list = []
    model.eval()
    criterion = nn.MSELoss()
    last_name=all_data[0][-2]
    with torch.no_grad():
        for i in range (len(all_data)-args.in_len-args.out_len+1):
            num=all_data[i][-2]
            num_after=all_data[i+args.in_len+args.out_len-1][-2]
            # print(i,num,num_after)
            if num_after!=num: 
                continue
            if (num!=last_name):
                # pred_list=np.concatenate(pred_list, axis=0)
                # true_list=np.concatenate(true_list, axis=0)
                # print(pred_list.shape,true_list.shape)
                pred_list=np.array(pred_list)
                true_list=np.array(true_list)
                mae, mse, rmse, mape, mspe,plcc,srcc=metric(pred_list, true_list)
                plot(true_list[:,-1],pred_list[:,-1],f"{last_name}")
                print(f"video_id:{last_name} ,mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")
                pred_list = []
                true_list = []
                last_name=num
            x=all_data[i:i+args.in_len,1]
            x=scaler.transform(x.reshape(-1,1)).reshape(1,-1)
            y=all_data[i+args.in_len:i+args.in_len+args.out_len,1].squeeze()
            # print(x.shape,y.shape,x,y)
            pred=model(torch.from_numpy(x).float().to(args.device)).detach().cpu().numpy()
            pred=inverse(pred).squeeze()
            # print(pred.shape,y.shape,pred,y)
            x_data=inverse(x.squeeze()).squeeze()
            # pred=adjust_pred(x_data)
            # pred=predict_next_sequence(x_data)
            tmp=pearsonr(y, pred)[0]
            print(i,x_data,y,pred,tmp)
            pred_list.append(pred)
            true_list.append(y)
    
    return mae, mse, rmse, mape, mspe,plcc,srcc

def train(args,train_loader,test_loader,model):
    model.train()
    time_now = time.time()
    train_steps = len(train_loader)
    model_optim = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    custom_loss = CustomLoss()

    for epoch in range(args.epochs):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            model.train()
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)

            outputs = model(batch_x)
            loss1 = criterion(outputs, batch_y)
            loss2=1-calculate_plcc(batch_y,outputs)

            loss3=custom_loss(outputs, batch_y, batch_x)
            loss=loss1+loss2+loss3
            train_loss.append(loss.item())
            # print(batch_y,outputs)
            loss.backward()
            model_optim.step()

            if (i + 1) % 100 == 0:
                train_PLCC=PLCC(outputs.detach().cpu().numpy(),batch_y.detach().cpu().numpy())
                logging.info(f"Epoch: {epoch + 1}, Iter: {i + 1}/{train_steps}, Loss1: {loss1.item():.7f}, Loss2: {loss2.item():.7f}, Loss: {loss.item():.7f}, PLCC: {train_PLCC:.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.epochs - epoch) * train_steps - i)
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

                # mae, mse, rmse, mape, mspe,plcc,srcc = test(args,test_loader,model)
                # logging.info(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")

        logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        mae, mse, rmse, mape, mspe,plcc,srcc = test(args,test_loader,model)
        logging.info(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")

        torch.save(model.state_dict(), f"./checkpoints/model_DNN_{args.tag}_inlen_{args.in_len}_outlen_{args.out_len}_fealen_{args.fea_len}_epoch_{epoch}.pth")
    return model

def main(args):
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output_100000.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./Time-Series-Library/dataset/bandwidth/bandwidth.csv")

    # all_data,train_idx,test_idx=split(args,"./dataset/output_100000.csv")
    if (args.tag=="10w"): ori_data,all_data,train_idx,test_idx=split(args,"./dataset/gtscore_10w.csv")
    else: ori_data,all_data,train_idx,test_idx=split(args,"./dataset/gtscore.csv")
    train_loader=CustomDataset(all_data,train_idx,args)
    test_loader=CustomDataset(all_data,test_idx,args)
    train_loader=DataLoader(train_loader,batch_size=args.batch,shuffle=True)
    test_loader=DataLoader(test_loader,batch_size=args.batch,shuffle=False)

    if (args.fea_len==1): model=DNNModel(args.in_len,1,args.out_len).to(args.device)
    else: model=DNNModel(args.in_len,all_data.shape[1],args.out_len).to(args.device)
    model.eval()
    if (args.mode==1): model=train(args,train_loader,test_loader,model)

    model.load_state_dict(torch.load(f"./checkpoints/model_DNN_{args.tag}_inlen_{args.in_len}_outlen_{args.out_len}_fealen_{args.fea_len}_epoch_{98}.pth",weights_only=True))
    model=model.to(args.device)
    mae, mse, rmse, mape, mspe,plcc,srcc=test_video(args,ori_data,model)
    print(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple train function with args')
    parser.add_argument('-mode', type=int, default=1, help='in len')
    parser.add_argument('-in_len', type=int, default=5, help='in len')
    parser.add_argument('-out_len', type=int, default=5, help='in len')
    parser.add_argument('-fea_len', type=int, default=1, help='in len')
    parser.add_argument('-batch', type=int, default=32, help='in len')
    parser.add_argument('-epochs', type=int, default=10, help='in len')
    parser.add_argument('-lr', type=float, default=5e-4, help='in len')
    parser.add_argument('-device', type=str, default="cuda", help='in len')
    parser.add_argument('-tag', type=str, default="all", help='in len')

    args = parser.parse_args()
    main(args)
