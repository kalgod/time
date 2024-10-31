import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import time
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os

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
        res.append(pearsonr(a, b)[0])
        print(i,a,b,res[-1],"\n")
    return np.mean(res)

def SRCC(pred, true):
    res=[]
    for i in range (len(pred)):
        a=pred[i].squeeze()
        b=true[i].squeeze()
        res.append(spearmanr(a, b)[0])
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
    np.random.shuffle(all_idx)
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

class DNNModel(nn.Module):
    def __init__(self, in_seq, n, out_seq):
        super(DNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_seq * n, 64)  # 第一层全连接层
        self.fc2 = nn.Linear(64, 32)          # 第二层全连接层
        self.fc3 = nn.Linear(32, out_seq)     # 输出层
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))           # 使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x=torch.relu(x)
        # x=self.sigmoid(x)
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
                print(last_name,pred_list.shape,true_list.shape)
                plot(true_list,pred_list,f"{last_name}")
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
            pred_list.append(pred)
            true_list.append(y)
    
    return mae, mse, rmse, mape, mspe,plcc,srcc

def train(args,train_loader,test_loader,model):
    model.train()
    time_now = time.time()
    train_steps = len(train_loader)
    model_optim = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

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
            loss=loss1+loss2
            train_loss.append(loss.item())
            # print(batch_y,outputs)
            loss.backward()
            model_optim.step()

            if (i + 1) % 100 == 0:
                train_PLCC=PLCC(outputs.detach().cpu().numpy(),batch_y.detach().cpu().numpy())
                print(f"Epoch: {epoch + 1}, Iter: {i + 1}/{train_steps}, Loss1: {loss1.item():.7f}, Loss2: {loss2.item():.7f}, Loss: {loss.item():.7f}, PLCC: {train_PLCC:.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.epochs - epoch) * train_steps - i)
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

                mae, mse, rmse, mape, mspe,plcc,srcc = test(args,test_loader,model)
                print(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        # mae, mse, rmse, mape, mspe,plcc,srcc = test(args,test_loader,model)
        # print(f"Train Loss: {train_loss:.7f} Test Loss: {mse:.7f} Test RMSE: {rmse:.7f} Test MAPE: {mape:.7f} Test PLCC: {plcc:.7f} Test SRCC: {srcc:.7f}")
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
    # model=train(args,train_loader,test_loader,model)

    model.load_state_dict(torch.load(f"./checkpoints/model_DNN_{args.tag}_inlen_{args.in_len}_outlen_{args.out_len}_fealen_{args.fea_len}_epoch_{9}.pth",weights_only=True))
    model=model.to(args.device)
    test_video(args,ori_data,model)
    print(f"mae: {mae:.7f}, mse: {mse:.7f}, rmse: {rmse:.7f}, mape: {mape:.7f}, mspe: {mspe:.7f}, plcc: {plcc:.7f}, srcc: {srcc:.7f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple train function with args')
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
