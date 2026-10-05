1、
 残差定义：$e_i = y_i-\hat y_i = y_i-\hat\beta_0-\hat\beta_1 x_i$
OLS一阶条件第一个方程：
$$
\frac{\partial RSS}{\partial \beta_0} = -2\sum_{i=1}^n\left(y_i-\beta_0-\beta_1 x_i\right)=0
$$
OLS估计值代入，残差 $e_i = y_i-\hat\beta_0-\hat\beta_1 x_i$
$$
\sum_{i=1}^n \left(y_i-\hat\beta_0-\hat\beta_1 x_i\right)=\sum_{i=1}^n e_i = \boldsymbol{0}
$$
第二个一阶偏导条件：
$$
\frac{\partial RSS}{\partial \beta_1} = -2\sum_{i=1}^n x_i\left(y_i-\beta_0-\beta_1 x_i\right)=0
$$
带入OLS估计量，残差 $e_i = y_i-\hat\beta_0-\hat\beta_1 x_i$
$$
\sum_{i=1}^n x_i\left(y_i-\hat\beta_0-\hat\beta_1 x_i\right)=\sum_{i=1}^n x_i e_i=\boldsymbol{0}
$$
2、
$R^2$
 $R^2=1-\frac{RSS}{TSS}$，TSS总平方和只跟响应变量y有关，固定不变。
新增一个完全随机无关变量，样本层面多少会解释一点点噪声，RSS会小幅下降
$RSS$变小，则 $1-\frac{RSS}{TSS}$，**$R^2$一定会上升。
$$
R_{adj}^2 = 1-\frac{RSS/(n-k-1)}{TSS/(n-1)}
$$
- $n$样本量，$k$自变量个数；增加变量，$k$增大，分母自由度 $n-k-1$减小。
- 虽然RSS轻微下降，但惩罚项（自由度）起作用。如果新增变量真实没有解释力，自由度惩罚占上风，$R_{adj}^2$会下降。

普通$R^2$没有自由度惩罚，只要增加自变量，哪怕是随机垃圾变量，$R^2$也倾向上涨；调整$R^2_{adj}$对增加自变量做自由度惩罚，只有新变量带来足够RSS下降，调整R方才会提高。
