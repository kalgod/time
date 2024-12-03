import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义高斯变体函数
def gaussian_variant(x, a, b, mu, sigma):
    """高斯分布变体模型"""
    return a + b * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# 给定的部分序列 x，代表数据的某一部分
# 调整此输入序列作为示例，确保它不是完整的高斯分布
x_data = np.array([ 0.9, 0.7, 0.4, 0.2])  # 原始部分序列

# 生成 x 的坐标
x_coords = np.arange(len(x_data))

# 估计初始参数
mu = np.mean(x_data)  # 均值
sigma = np.std(x_data)  # 标准差
initial_guess = [np.min(x_data), 1, mu, sigma]  # a, b, mu, sigma 的初值

# 拟合高斯分布变体
params, _ = curve_fit(gaussian_variant, x_coords, x_data, p0=initial_guess)

# 提取拟合的参数
a, b, mu, sigma = params
print(f"拟合得到的参数: a={a}, b={b}, mu={mu}, sigma={sigma}")

# 生成未来的结果
future_n = 5  # 预测未来 5 个值
future_coords = np.arange(len(x_data), len(x_data) + future_n)

# 使用拟合后的高斯函数预测未来的值
future_values = gaussian_variant(future_coords, *params)

# 合并当前和预测值
all_values = np.concatenate((x_data, future_values))

# 绘图展示
plt.figure(figsize=(10, 6))
plt.plot(x_coords, x_data, 'bo-', label='ori')
plt.plot(future_coords, future_values, 'ro-', label='pred')
plt.plot(np.arange(len(all_values)), gaussian_variant(np.arange(len(all_values)), *params), 'g--', label='gau')
plt.xlabel('序列索引')
plt.ylabel('值')
plt.xticks(np.arange(len(all_values)))  # 确保显示所有索引
plt.legend()
plt.title('高斯分布变体的拟合与预测')
plt.grid()
plt.savefig('gaussian_variant_fit.png')