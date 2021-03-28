import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
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
from sklearn import model_selection
hit = pd.read_csv("Hitters.csv")
df = hit.copy()
df = df.dropna()
df.head()


print(df.info())
print(df.describe().T)

dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
print(dms.head())

y = df["Salary"]
X_ = df.drop(["Salary","League","Division","NewLeague"], axis = 1).astype("float64")

print(X_.head())

X = pd.concat([X_, dms[["League_N", "Division_W","NewLeague_N"]]], axis = 1)

print(X_.head())

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)

print("X_train", X_train.shape)

print("y_train",y_train.shape)

print("X_test",X_test.shape)

print("y_test",y_test.shape)

training = df.copy()

print("training", training.shape)


pca = PCA()

X_reduced_train = pca.fit_transform(scale(X_train))


print(X_reduced_train[0:1,:])
print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)[0:5])

lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train, y_train)

print(pcr_model.intercept_)
print(pcr_model.coef_)

y_pred = pcr_model.predict(X_reduced_train)

print(y_pred[0:5])
print(np.sqrt(mean_squared_error(y_train, y_pred)))
print(df["Salary"].mean())
print(r2_score(y_train, y_pred))
pca2 = PCA()
X_reduced_test = pca2.fit_transform(scale(X_test))
y_pred = pcr_model.predict(X_reduced_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))



lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:10], y_train)
y_pred = pcr_model.predict(X_reduced_test[:,0:10])
print(np.sqrt(mean_squared_error(y_test, y_pred)))

cv_10 = model_selection.KFold(n_splits = 10,
                             shuffle = True,
                             random_state = 1)
lm = LinearRegression()
RMSE = []

for i in np.arange(1, X_reduced_train.shape[1] + 1):
    score = np.sqrt(-1 * model_selection.cross_val_score(lm,
                                                         X_reduced_train[:, :i],
                                                         y_train.ravel(),
                                                         cv=cv_10,
                                                         scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Maaş Tahmin Modeli İçin PCR Model Tuning');

lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:6], y_train)
y_pred = pcr_model.predict(X_reduced_train[:,0:6])
print(np.sqrt(mean_squared_error(y_train, y_pred)))
y_pred = pcr_model.predict(X_reduced_test[:,0:6])
print(np.sqrt(mean_squared_error(y_test, y_pred)))



