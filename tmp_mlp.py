import csv
import numpy as np
import tensorflow as tf
import time
# import matplotlib.pyplot as plt

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epochs = 10
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0 or epoch == self.epochs - 1:  # 每10个epoch或最后一个epoch显示一次
            elapsed_time = time.time() - self.start_time
            eta = elapsed_time / (epoch + 1) * (self.epochs - epoch - 1)
            print(f'\rEpoch {epoch+1}/{self.epochs} - '
                  f'loss: {logs["loss"]:.4f} - '
                  f'val_loss: {logs["val_loss"]:.4f} - '
                  f'ETA: {eta:.0f}s', end='')

def load_data(file_path):
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            try:
                net_cost = float(row[19])
                downloaded_bytes = float(row[20])
                speed = float(row[27])
                speed1 = float(row[28])
                if (row[28]==np.nan): continue
                if (speed1==0): continue
                clienttime = float(row[29])
                data.append([net_cost, downloaded_bytes, speed, speed1, clienttime])
            except ValueError:
                continue
    return np.array(data)

def prepare_data(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back].flatten())
        y.append(data[i+look_back, 3])  # speed1 is at index 3
    return np.array(X), np.array(y)

def train_test_split(X, y, test_size=0.2):
    n = len(X)
    test_n = int(n * test_size)
    indices = np.random.permutation(n)
    test_indices = indices[:test_n]
    train_indices = indices[test_n:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)  # 添加小量以避免除零
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X * self.std + self.mean

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_tot)

def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nmse = mse / (np.max(y_true) - np.min(y_true))**2
    r2 = r2_score(y_true, y_pred)
    mae_gt = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8))
    return mse, mae, rmse, nmse, r2, mae_gt

def main(file_path):
    print("正在加载数据...")
    data = load_data(file_path)
    if len(data) == 0:
        print("没有有效数据。请检查CSV文件。")
        return
    
    print("正在准备数据...")
    X, y = prepare_data(data)

    print("正在分割数据...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("正在标准化数据...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print("正在构建模型...")
    model = build_model(X_train_scaled.shape[1])

    print("开始训练模型...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=1,
        batch_size=64,
        validation_split=0.2,
        verbose=0,
        callbacks=[CustomCallback()]
    )

    print("\n训练完成。")

    print("正在进行预测...")
    y_pred_scaled = model.predict(X_test_scaled).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    print("正在评估模型...")
    mse, mae, rmse, nmse, r2,mae_gt = evaluate_predictions(y_test, y_pred)
    # gt_mean = np.mean(y_test)
    # mae_gt = mae / gt_mean
    print("\nMLP Model Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE/gt: {mae_gt:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE (Relative MSE): {rmse:.4f}")
    print(f"NMSE (Normalized MSE): {nmse:.4f}")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(history.history['loss'])
    print(history.history['val_loss'])
    # # 绘制训练和验证损失
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    file_path = 'dataset/all_bw.csv'  # 替换为您的CSV文件路径
    main(file_path)