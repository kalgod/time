import numpy as np
from scipy.stats import pearsonr, spearmanr


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
    corr = CORR(pred, true)
    plcc = PLCC(pred, true)
    srcc = SRCC(pred, true)

    return mae, mse, rmse, mape, mspe,plcc,srcc