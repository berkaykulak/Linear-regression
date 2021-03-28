from warnings import filterwarnings
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

filterwarnings('ignore')

ad = pd.read_csv("Advertising.csv", usecols = [1,2,3,4])
df = ad.copy()



sns.pairplot(df, kind  ="reg");
sns.jointplot(x = "TV", y = "sales", data = df, kind = "reg")

X = df[["TV"]]

X = sm.add_constant(X)


y = df["sales"]


lm = sm.OLS(y,X)
model = lm.fit()


lm = smf.ols("sales ~ TV", df)

model = lm.fit()




g = sns.regplot(df["TV"], df["sales"], ci=None, scatter_kws={'color':'r', 's':9})
g.set_title("Model Denklemi: Sales = 7.03 + TV*0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")

plt.xlim(-10,310)
plt.ylim(bottom=0);

X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X, y)

X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X, y)

print(model.predict([[30]]))

yeni_veri = [[5],[90],[200]]

print(model.predict(yeni_veri))













