import numpy as np
import pandas as pd

data = pd.read_csv(
    "ColorHistogram.asc",
    header=None,
    delim_whitespace=True,
    index_col=0,
    dtype=np.float64,
)

# 计算数据的均值
mean = np.mean(data, axis=0)

# 将数据进行中心化
data_centered = data - mean

# 计算协方差矩阵
cov_matrix = np.cov(data_centered.T)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 将特征向量按照特征值从大到小排序
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# 取出前 5 个特征向量
n_components = 5
selected_eigenvectors = sorted_eigenvectors[:, :n_components]

pca_data = np.dot(data_centered, selected_eigenvectors)

# 计算 PCA 之前和 PCA 之后数据的方差
variance_before = np.var(data, axis=0).sum()
variance_after = np.var(pca_data, axis=0).sum()

# 输出结果
print("PCA 之前数据方差：", variance_before)
print("PCA 之后数据（降至%d维）：\n" % n_components, pca_data)
print("PCA 之后数据方差：", variance_after)
