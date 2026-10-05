# carseats_analysis.py
import statsmodels.api as sm
from statsmodels.datasets import carseats
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = carseats.load_pandas().data
# 因变量 Sales 销售额；自变量 Price, Income, Advertising, ShelveLoc(分类)
y = data["Sales"]
X_raw = data[["Price","Income","Advertising","ShelveLoc"]]
# 生成虚拟变量，ShelveLoc转为哑变量，statsmodels自动处理分类变量
X = pd.get_dummies(X_raw, columns=["ShelveLoc"], drop_first=True)
# 添加常数项beta0
X = sm.add_constant(X)

# 构建线性回归模型
model = sm.OLS(y,X).fit()
print("=======回归拟合报告=======")
print(model.summary())

# 找出基准组：get_dummies drop_first=True，被删掉那一列就是基准
# ShelveLoc有三个取值：Bad,Good,Medium；如果删掉ShelveLoc_Bad，则基准组=Bad；
# 如果删掉Medium基准组是Medium
print("\n====ShelveLoc基准组说明====")
# 查看X列名
print("X变量列名：",X.columns.tolist())
# ShelveLoc[Good]系数
print("\n====ShelveLoc[Good]系数商业解读====")
print("ShelveLocGood的系数含义：保持Price、Income、Advertising不变，货架位置为Good对比基准组，销售额Sales平均变化量。")

vif_df = pd.DataFrame()
vif_df["feature"] = X.columns
vif_values = []
for idx in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, idx)
    vif_values.append(vif)
vif_df["VIF"] = vif_values
print("\n====各变量VIF方差膨胀因子=====")
print(vif_df)
# VIF经验阈值：VIF>5~10认为存在比较严重多重共线性风险
