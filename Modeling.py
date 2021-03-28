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


print(df.head())
print(df.info())

print(df.describe().T)

print(df.isnull().values.any())
print(df.corr())

sns.pairplot(df, kind  ="reg");
sns.jointplot(x = "TV", y = "sales", data = df, kind = "reg")

X = df[["TV"]]
print(X[0:5])
X = sm.add_constant(X)
print(X[0:5])

y = df["sales"]
print(y[0:5])

lm = sm.OLS(y,X)
model = lm.fit()
print(model.summary())

lm = smf.ols("sales ~ TV", df)

model = lm.fit()
print(model.summary())

print(model.params)
print(model.summary().tables[1])
print(model.conf_int())
print(model.f_pvalue)
print("f_pvalue: ", "%.4f" % model.f_pvalue)
print("fvalue: ", "%.2f" % model.fvalue)
print("tvalue: ", "%.2f" % model.tvalues[0:1])



print(model.rsquared_adj)
print(model.fittedvalues[0:5])
print(y[0:5])
print("Sales = " +  str("%.2f" % model.params[0]) + " + TV" + "*" + str("%.2f" % model.params[1]))
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

print(model.intercept_)
print(model.coef_)

print(model.score(X,y))
print(model.predict(X)[0:10])





















