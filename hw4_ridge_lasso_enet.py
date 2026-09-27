import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

hitters_raw_text = """Salary,AtBat,Hits,HmRun,Runs,RBI,Walks,Years,CAtBat,CHits,CHmRun,CRuns,CRBI,CWalks,League,Division,PutOuts,Assists,Errors,NewLeague
325,315,81,16,44,49,34,2,315,81,16,44,49,34,N,W,632,43,10,N
100,479,130,10,55,39,22,1,479,130,10,55,39,22,A,W,190,11,7,N
750,496,141,17,74,65,40,14,5338,1457,195,835,722,388,A,E,845,82,8,A
1000,321,86,10,39,42,30,3,321,86,10,39,42,30,N,E,249,3,1,A
517,594,169,16,84,78,51,11,5702,1593,157,794,764,429,A,W,437,10,4,N
250,385,101,8,40,44,25,2,385,101,8,40,44,25,N,W,177,27,9,N
550,523,142,12,66,67,37,11,5164,1410,143,722,679,372,A,E,916,96,12,A
800,498,139,29,78,88,54,10,4730,1329,186,734,753,452,A,E,1053,74,14,A
120,179,45,3,23,20,14,1,179,45,3,23,20,14,A,W,122,10,4,N
90,194,47,1,22,14,11,1,194,47,1,22,14,11,N,E,119,11,2,N
775,600,173,18,91,72,76,12,5360,1513,166,775,683,560,A,W,267,42,10,A
110,183,39,3,20,21,22,2,268,63,4,28,25,28,N,E,80,11,3,N
850,538,151,16,77,70,48,10,4680,1307,129,641,591,432,A,E,78,32,9,A
140,439,118,12,61,60,37,3,1041,271,24,130,122,90,A,W,137,19,10,N
725,616,174,18,95,78,60,15,6106,1708,174,868,820,600,A,W,128,16,8,A
130,264,69,7,34,37,21,2,264,69,7,34,37,21,N,W,88,14,4,N
100,277,68,5,34,24,22,1,277,68,5,34,24,22,N,E,114,2,4,N
123,300,79,13,37,31,24,2,300,79,13,37,31,24,N,E,45,11,3,N
149,225,60,11,30,31,26,1,225,60,11,30,31,26,N,W,112,7,3,N
300,520,142,13,76,63,42,7,3080,843,67,395,324,227,A,E,37,23,6,A
"""

from io import StringIO
hitters = pd.read_csv(StringIO(hitters_raw_text))
hitters = hitters.dropna()

X_raw = hitters.drop("Salary", axis=1)
X_raw = pd.get_dummies(X_raw, drop_first=True)
y = hitters["Salary"]


X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

alphas = np.logspace(-3, 4, 100)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
ridge_pred = ridge_cv.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=100000)
lasso_cv.fit(X_train, y_train)
lasso_pred = lasso_cv.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)

enet_cv = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=5, max_iter=100000)
enet_cv.fit(X_train, y_train)
enet_pred = enet_cv.predict(X_test)
enet_mse = mean_squared_error(y_test, enet_pred)

#输出结果
print("==== Ridge ====")
print(f"最优alpha: {ridge_cv.alpha_:.4f}")
print(f"测试集MSE: {ridge_mse:.2f}")

print("\n==== Lasso ====")
print(f"最优alpha: {lasso_cv.alpha_:.4f}")
print(f"测试集MSE: {lasso_mse:.2f}")

print("\n==== ElasticNet ====")
print(f"最优alpha: {enet_cv.alpha_:.4f}")
print(f"测试集MSE: {enet_mse:.2f}")