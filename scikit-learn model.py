from warnings import filterwarnings
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

ad = pd.read_csv("Advertising.csv", usecols = [1,2,3,4])
df = ad.copy()



X = df.drop("sales", axis = 1)
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)


lm = sm.OLS(y_train, X_train)
model = lm.fit()

lm = LinearRegression()
model = lm.fit(X_train, y_train)
print(model.intercept_)
print(model.coef_)

yeni_veri = [[30], [10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

print(model.predict(yeni_veri))
rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))

print(rmse)

rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print(rmse)









