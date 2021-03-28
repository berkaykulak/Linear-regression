from warnings import filterwarnings
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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



yeni_veri = [[5],[90],[200]]



lm = smf.ols("sales ~ TV", df)
model = lm.fit()

mse = mean_squared_error(y, model.fittedvalues)

print(mse)

rmse = np.sqrt(mse)

print(rmse)

print(reg.predict(X)[0:10])
print(y[0:10])

k_t = pd.DataFrame({"gercek_y": y[0:10],
                   "tahmin_y": reg.predict(X)[0:10]})
print(k_t)

k_t["hata"] = k_t["gercek_y"] - k_t["tahmin_y"]

print(k_t)
k_t["hata_kare"] = k_t["hata"]**2
print(k_t)
print(np.sum(k_t["hata_kare"]))
print(np.mean(k_t["hata_kare"]))
print(np.sqrt(np.mean(k_t["hata_kare"])))
print(model.resid[0:10])
plt.plot(model.resid)

