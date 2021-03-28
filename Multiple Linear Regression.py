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

print(df.head())

X = df.drop("sales", axis = 1)
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
training = df.copy()
print(training.shape)




