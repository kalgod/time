import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import time

import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from momentfm import MOMENTPipeline

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
    all_data=all_data[:,1:]
    all_data=scaler.fit_transform(all_data)
    # print(all_data.shape,all_data[:2])
    res=[]
    for i in range (len(all_data)-args.in_len-args.out_len+1):
        num=all_data[i][-2]
        num_after=all_data[i+args.in_len+args.out_len-1][-2]
        # print(i,num,num_after)
        if num_after<=num: continue
        res.append(i)
    res=np.array(res)
    all_data=np.array(all_data,dtype=np.float32)
    return all_data,res

def split(args,file_name):
    all_data,all_idx=generate(args,file_name)
    np.random.shuffle(all_idx)
    train_len=int(len(all_idx)*0.7)
    train_idx = all_idx[:train_len]
    test_idx = all_idx[train_len:]
    print(len(all_idx),len(train_idx),len(test_idx))
    all_data=torch.from_numpy(all_data).float()
    return all_data,train_idx,test_idx

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
        mask=np.ones(args.in_len)
        return x,y,mask

class InformerDataset:
    def __init__(
        self,
        forecast_horizon: int = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        args=None
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = args.in_len
        self.forecast_horizon = args.out_len
        self.full_file_path_and_name = "./dataset/2fea_100000.csv"
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        # Read data
        self._read_data()

    def _get_borders(self):
        n_train = 12 * 30 * 24
        n_val = 4 * 30 * 24
        n_test = 4 * 30 * 24

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects().interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "test":
            self.data = df[data_splits[1], :]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T

            return timeseries, input_mask

    def __len__(self):
        if self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1

def main(args):
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output_100000.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./Time-Series-Library/dataset/bandwidth/bandwidth.csv")

    # Load data
    train_dataset = InformerDataset(data_split="train", random_seed=13, forecast_horizon=args.out_len,args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    test_dataset = InformerDataset(data_split="test", random_seed=13, forecast_horizon=args.out_len,args=args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            "n_channels": 1,
            'forecast_horizon': args.out_len,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )

    model.init()
    # print("Unfrozen parameters:")
    # for name, param in model.named_parameters():    
    #     if param.requires_grad:
    #         print('    ', name)

    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=13) 

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cur_epoch = 0
    max_epoch = args.epochs

    # Move the model to the GPU
    model = model.to(device)

    # Move the loss function to the GPU
    criterion = criterion.to(device)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create a OneCycleLR scheduler
    max_lr = 1e-4
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

    # Gradient clipping value
    max_norm = 5.0

    while cur_epoch < max_epoch:
        losses = []
        for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)
            with torch.cuda.amp.autocast():
                output = model(timeseries, input_mask)

            loss = criterion(output.forecast, forecast)

            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

        # Step the learning rate scheduler
        scheduler.step()
        cur_epoch += 1
        
        # Evaluate the model on the test split
        trues, preds, histories, losses = [], [], [], []
        model.eval()
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
            # Move the data to the GPU
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)
                
                loss = criterion(output.forecast, forecast)                
                losses.append(loss.item())

                trues.append(forecast.detach().cpu().numpy())
                preds.append(output.forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())
        
        losses = np.array(losses)
        average_loss = np.average(losses)
        model.train()

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)
        
        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

        print(f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f} | Test MAE/GT: {metrics.mape:.3f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple train function with args')
    parser.add_argument('-in_len', type=int, default=5, help='in len')
    parser.add_argument('-out_len', type=int, default=5, help='in len')
    parser.add_argument('-fea_len', type=int, default=1, help='in len')
    parser.add_argument('-batch', type=int, default=32, help='in len')
    parser.add_argument('-epochs', type=int, default=10, help='in len')
    parser.add_argument('-lr', type=float, default=5e-4, help='in len')
    parser.add_argument('-device', type=str, default="cuda", help='in len')

    args = parser.parse_args()
    main(args)
