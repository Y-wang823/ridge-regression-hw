import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号乱码
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

# 读取Netflix数据集
df = pd.read_csv("netflix_titles.csv", on_bad_lines='skip')

# 数据清洗：只保留电影，预测片长
df = df[df["type"] == "Movie"].copy()
df["duration"] = df["duration"].str.replace(" min", "").astype(float)
df = df.dropna(subset=["release_year", "rating", "country", "duration"])

# 特征X，预测目标y（电影时长）
X = df[["release_year", "rating", "country"]]
y = df["duration"]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 区分数值特征、类别特征
cat_features = ["rating", "country"]
num_features = ["release_year"]

# 预处理流水线：标准化+独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ]
)

# ElasticNetCV自动交叉验证，寻找最优alpha、l1_ratio
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1,0.3,0.5,0.7,0.9],
    alphas=np.logspace(-3,1,20),
    max_iter=5000,
    random_state=42
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", elastic_cv)
])

# 训练模型
pipe.fit(X_train, y_train)

# 预测
y_pred = pipe.predict(X_test)

# 评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("======= ElasticNet Netflix 分析结果 =======")
print(f"最优 l1_ratio: {pipe.named_steps['model'].l1_ratio_:.2f}")
print(f"最优 alpha: {pipe.named_steps['model'].alpha_:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# 绘图：真实值 vs 预测值
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("真实片长(min)")
plt.ylabel("预测片长(min)")
plt.title("ElasticNet Netflix电影时长预测")
plt.show()
