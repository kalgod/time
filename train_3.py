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
from torchvision import datasets, transforms
from PIL import Image
import logging
import sys
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import json
import pickle

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
    # all_data=remove_adjacent_duplicates(all_data)
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
    all_data=all_data[:,-3].reshape(-1,1)
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
    with open('./dataset/video_data.pkl', 'rb') as pickle_file:
        video_data = pickle.load(pickle_file)
    # video_data={}
    # image_transform= transforms.Compose([
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])
    # video_list=os.listdir("./dataset/youtube")
    # video_list=sorted(video_list)
    # for video_id in tqdm(video_list):
    #     video_data[video_id]={}
    #     metainfo=ori_data[ori_data[:,-2]==video_id][0,-1]
    #     video_data[video_id]["metainfo"]=metainfo
    #     video_data[video_id]["image"]=[]
    #     image_path=f"./dataset/youtube/{video_id}/frames"
    #     image_len=len(os.listdir(image_path))
    #     video_data[video_id]["image_len"]=image_len
    #     for i in range (image_len):
    #         cur_path=f"./dataset/youtube/{video_id}/frames/{i}.png"
    #         image=Image.open(cur_path)
    #         image=image_transform(image)*255
    #         # image=np.array(image)
    #         # image=torch.from_numpy(image).float()
    #         # print(image,image.shape)
    #         # print(f"image size:{image.shape}")
    #         video_data[video_id]["image"].append(image)
    #     video_data[video_id]["image"]=np.stack(video_data[video_id]["image"])
    #     logging.info(f"video_id:{video_id},image:{video_data[video_id]['image'].shape},metainfo:{metainfo}")
    # # 存储字典到 Pickle 文件
    # with open('./dataset/video_data.pkl', 'wb') as pickle_file:
    #     pickle.dump(video_data, pickle_file)
    # exit(0)
    return ori_data,all_data,train_idx,test_idx,video_data

class CustomDataset(Dataset):
    def __init__(self, ori_data,all_data,all_idx,args,video_data):
        self.all_data = all_data
        self.all_idx = all_idx
        self.args=args
        self.ori_data=ori_data
        self.video_data=video_data
        self.transform= transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    def __len__(self):
        return len(self.all_idx)

    def __getitem__(self, idx):
        idx=self.all_idx[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_idx=self.ori_data[idx:idx+self.args.in_len,0]
        video_id=self.ori_data[idx,-2]
        metainfo=self.ori_data[idx,-1]
        x=self.all_data[idx:idx+self.args.in_len]
        y=self.all_data[idx+self.args.in_len:idx+self.args.in_len+self.args.out_len]
        if (args.fea_len==1):
            x=x[:,-1]
        y=y[:,-1]
        # images=self.video_data[video_id]["image"][list(image_idx)]
        # images=torch.from_numpy(images).float()
        # metainfo=self.video_data[video_id]["metainfo"]

        # print(idx,image_idx,video_id,metainfo,x,y)
        # images=[]
        # for i in range (len(image_idx)):
        #     image_path=f"./dataset/youtube/{video_id}/frames/{image_idx[i]}.png"
        #     image=Image.open(image_path)
        #     image=self.transform(image)*255
        #     # image=np.array(image)
        #     # image=torch.from_numpy(image).float()
        #     # print(image,image.shape)
        #     # print(f"image size:{image.shape}")
        #     images.append(image)
        # images=torch.stack(images)

        # metainfo = metainfo.split("Title: ")[1].split(" Categories:")[0].strip()

        # print(metainfo)
        # logging.info(f"idx:{idx},image_idx:{image_idx},video_id:{video_id},metainfo:{metainfo},x:{x.shape},y:{y.shape},images:{images.shape}")
        return x,y,y,y

class CLIP:
    def __init__(self, device, model_name="openai/clip-vit-base-patch32"):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # normalize features
        return image_features

    def encode_text(self, text):
        inputs = self.processor(text=text, padding=True, return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # normalize features
        return text_features

class DNNModel(nn.Module):
    def __init__(self,input_size, output_size):
        super(DNNModel, self).__init__()
        self.clip=CLIP(device="cuda")
        # 使用 Hugging Face 加载 CLIP 模型和处理器
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 将 CLIP 模型的输出特征维度获取
        self.clip_feature_dim = self.clip.clip_model.config.projection_dim

        # 时序数据特征层
        self.fc_time = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU())  # 输入为一维时序特征
        self.fc_image = nn.Sequential(
            nn.Linear(self.clip_feature_dim*1,64),
            nn.ReLU())  # 输入为一维时序特征
        self.fc_text = nn.Sequential(
            nn.Linear(self.clip_feature_dim*1,64),
            nn.ReLU())  # 输入为一维时序特征
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64*1, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, time_sequences, images, texts):
        # 处理时序数据
        texts=list(texts)
        # for i in range (len(texts)): print(i,len(texts[i]))
        # time_sequences=time_sequences.unsqueeze(-1)
        # time_sequences=time_sequences.view(-1,1)
        
        time_features = self.fc_time(time_sequences)  # 扩展维度后输入到全连接层

        # batch_size, in_seq, channel, H, W = images.size()
        # images = images.view(batch_size * in_seq, channel, H, W)
        # image_features = self.clip.encode_image(images)
        # image_features=self.fc_image(image_features)
        # image_features=image_features.view(batch_size, in_seq, -1)
        # image_features=image_features.mean(dim=1)

        # text_features = self.clip.encode_text(texts)
        # text_features=self.fc_text(text_features)

        # print(f"image_features:{image_features.shape},text_features:{text_features.shape},time_features:{time_features.shape}")
        # combined_features = torch.cat((time_features,image_features,text_features), dim=1)
        combined_features=time_features
        
        output = self.fc(combined_features)
        # output = self.fc(time_sequences)
        
        return output

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
        vx = a - torch.mean(a)
        vy = b - torch.mean(b)
        cova=torch.sum(vx * vy)
        if (cova==0): plcc=cova
        else: plcc = cova / (torch.norm(vx) * torch.norm(vy))

        # covariance = torch.sum((a - a_mean) * (b - b_mean))
        # std_true = torch.norm(a - a_mean)
        # std_pred = torch.norm(b - b_mean)
        # plcc = covariance / (std_true * std_pred+0e-9)

        all_plcc.append(plcc)
    plcc=torch.mean(torch.stack(all_plcc))
    return plcc

def test(args,test_loader,model):
    pred_list = []
    true_list = []
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, (batch_x, batch_y,batch_image,batch_text) in enumerate(test_loader):
            batch_x = batch_x.to(args.device)
            batch_image = batch_image.to(args.device)
            # batch_text = batch_text.to(args.device)
            outputs = model(batch_x,batch_image,batch_text)

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
        for i, (batch_x, batch_y,batch_image,batch_text) in enumerate(train_loader):
            model.train()
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            batch_image = batch_image.to(args.device)
            # batch_text = batch_text.to(args.device)

            outputs = model(batch_x,batch_image,batch_text)
            outputs=outputs
            loss1 = criterion(outputs, batch_y)
            loss2=1-calculate_plcc(batch_y,outputs)

            loss3=custom_loss(outputs, batch_y, batch_x)
            loss=loss1+loss2
            # if (loss1<0.1): loss=loss+loss2
            train_loss.append(loss.item())
            # print(batch_y,outputs)
            loss.backward()
            model_optim.step()

            if (i + 1) % 100 == 0:
                train_PLCC=PLCC(outputs.detach().cpu().numpy(),batch_y.detach().cpu().numpy())
                logging.info(f"Epoch: {epoch}, Iter: {i + 1}/{train_steps}, Loss1: {loss1.item():.7f}, Loss2: {loss2.item():.7f}, Loss: {loss.item():.7f}, PLCC: {train_PLCC:.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.epochs - epoch) * train_steps - i)
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

                # mae, mse, rmse, mape, mspe,plcc,srcc = test(args,test_loader,model)
                # logging.info(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")

        logging.info("Epoch: {} cost time: {}".format(epoch, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        mae, mse, rmse, mape, mspe,plcc,srcc = test(args,test_loader,model)
        logging.info(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")

        torch.save(model.state_dict(), f"./checkpoints/model_DNN_{args.tag}_inlen_{args.in_len}_outlen_{args.out_len}_fealen_{args.fea_len}_epoch_{epoch}.pth")
    return model

def main(args):
    # 设置 logging 配置
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO
        format='%(message)s',  # 日志格式
        handlers=[
            logging.FileHandler(args.log,mode='w'),  # 将日志输出到 log.txt 文件
            # logging.StreamHandler(sys.stdout)  # 也输出到控制台
        ]
    )
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output_100000.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./Time-Series-Library/dataset/bandwidth/bandwidth.csv")

    # all_data,train_idx,test_idx=split(args,"./dataset/output_100000.csv")
    ori_data,all_data,train_idx,test_idx,video_data=split(args,f"./dataset/gtscore_{args.tag}.csv")
    train_loader=CustomDataset(ori_data,all_data,train_idx,args,video_data)
    test_loader=CustomDataset(ori_data,all_data,test_idx,args,video_data)
    train_loader=DataLoader(train_loader,batch_size=args.batch,shuffle=True)
    test_loader=DataLoader(test_loader,batch_size=args.batch,shuffle=False)

    model=DNNModel(args.in_len,args.out_len).to(args.device)
    model.eval()
    if (args.mode==1): model=train(args,train_loader,test_loader,model)

    model.load_state_dict(torch.load(f"./checkpoints/model_DNN_{args.tag}_inlen_{args.in_len}_outlen_{args.out_len}_fealen_{args.fea_len}_epoch_{args.epochs-1}.pth",weights_only=True))
    model=model.to(args.device)
    mae, mse, rmse, mape, mspe,plcc,srcc = test(args,test_loader,model)
    logging.info(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")
    # mae, mse, rmse, mape, mspe,plcc,srcc=test_video(args,ori_data,model)
    # print(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")
    

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
    parser.add_argument('-tag', type=str, default="video", help='in len')
    parser.add_argument('-log', type=str, default="temp.txt", help='in len')

    args = parser.parse_args()
    main(args)
