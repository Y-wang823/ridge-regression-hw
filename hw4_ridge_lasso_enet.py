import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO

# Hitters棒球数据集文本
hitters_raw_text = """Salary,AtBat,Hits,HmRun,Runs,RBI,Walks,Years,CAtBat,CHits,CHmRun,CRuns,CRBI,CWalks,League,Division
325,315,81,16,44,49,34,2,315,81,16,44,49,34,N,W
100,479,130,10,55,39,22,1,479,130,10,55,39,22,A,W
750,496,141,17,74,65,40,14,5338,1457,195,835,722,388,A,E
1000,321,86,10,39,42,30,3,321,86,10,39,42,30,N,E
517,594,169,16,84,78,51,11,5702,1593,157,794,764,429,A,W
250,385,101,8,40,44,25,2,385,101,8,40,44,25,N,W
550,523,142,12,66,67,37,11,5164,1410,143,722,679,372,A,E
800,498,139,29,78,88,54,10,4730,1329,186,734,753,452,A,E
120,179,45,3,23,20,14,1,179,45,3,23,20,14,A,W
90,194,47,1,22,14,11,1,194,47,1,22,14,11,N,E
775,600,173,18,91,72,76,12,5360,1513,166,775,683,560,A,W
110,183,39,3,20,21,22,2,268,63,4,28,25,28,N,E
850,538,151,16,77,70,48,10,4680,1307,129,641,591,432,A,E
140,439,118,12,61,60,37,3,1041,271,24,130,122,90,A,W
725,616,174,18,95,78,60,15,6106,1708,174,868,820,600,A,W
130,264,69,7,34,37,21,2,264,69,7,34,37,21,N,W
100,277,68,5,34,24,22,1,277,68,5,34,24,22,N,E
123,300,79,13,37,31,24,2,300,79,13,37,31,24,N,E
149,225,60,11,30,31,26,1,225,60,11,30,31,26,N,W
300,520,142,13,76,63,42,7,3080,843,67,395,324,227,A,E
"""

# 读取文本数据
hitters = pd.read_csv(StringIO(hitters_raw_text))

# 数据预处理：删除缺失值，类别变量独热编码
hitters = hitters.dropna()
hitters = pd.get_dummies(hitters, drop_first=True)

# 划分特征X、目标y（预测薪资Salary）
X = hitters.drop("Salary", axis=1)
y = hitters["Salary"]

# 训练集、测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化（正则回归必须标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5折交叉验证训练三个正则模型
ridge = RidgeCV(cv=5)
lasso = LassoCV(cv=5)
enet = ElasticNetCV(cv=5)

ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)
enet.fit(X_train_scaled, y_train)

# 预测
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_enet = enet.predict(X_test_scaled)

# 评估指标
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

mse_enet = mean_squared_error(y_test, y_pred_enet)
r2_enet = r2_score(y_test, y_pred_enet)

# 打印结果
print("="*50)
print("Ridge岭回归")
print(f"最优alpha = {ridge.alpha_:.2f}")
print(f"测试集MSE = {mse_ridge:.2f}, R² = {r2_ridge:.3f}")
print("="*50)
print("Lasso回归")
print(f"最优alpha = {lasso.alpha_:.2f}")
print(f"测试集MSE = {mse_lasso:.2f}, R² = {r2_lasso:.3f}")
print("="*50)
print("ElasticNet弹性网")
print(f"最优alpha = {enet.alpha_:.2f}, 最优l1_ratio = {enet.l1_ratio_:.2f}")
print(f"测试集MSE = {mse_enet:.2f}, R² = {r2_enet:.3f}")
print("="*50)
