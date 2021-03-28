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
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
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

hit = pd.read_csv("Hitters.csv")
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)


ridge_model = Ridge(alpha = 0.1).fit(X_train, y_train)

print(ridge_model)
print(ridge_model.coef_)
print(10**np.linspace(10,-2,100)*0.5 )

lambdalar = 10 ** np.linspace(10, -2, 100) * 0.5

ridge_model = Ridge()
katsayilar = []

for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train, y_train)
    katsayilar.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdalar, katsayilar)
ax.set_xscale('log')

plt.xlabel('Lambda(Alpha) Değerleri')
plt.ylabel('Katsayılar/Ağırlıklar')
plt.title('Düzenlileştirmenin Bir Fonksiyonu Olarak Ridge Katsayıları');

y_pred = ridge_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))

lambdalar = 10**np.linspace(10,-2,100)*0.5

print(lambdalar[0:5])

ridge_cv = RidgeCV(alphas = lambdalar,
                   scoring = "neg_mean_squared_error",
                   normalize = True)

print(ridge_cv.fit(X_train, y_train))
print(ridge_cv.alpha_)

ridge_tuned = Ridge(alpha = ridge_cv.alpha_,
                   normalize = True).fit(X_train,y_train)
print(np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(X_test))))

hit = pd.read_csv("Hitters.csv")
df = hit.copy()
df = df.dropna()
ms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)
lasso_model = Lasso(alpha = 0.1).fit(X_train, y_train)
print(lasso_model)

print(lasso_model.coef_)

lasso = Lasso()
lambdalar = 10 ** np.linspace(10, -2, 100) * 0.5
katsayilar = []

for i in lambdalar:
    lasso.set_params(alpha=i)
    lasso.fit(X_train, y_train)
    katsayilar.append(lasso.coef_)

ax = plt.gca()
ax.plot(lambdalar * 2, katsayilar)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

print(lasso_model.predict(X_test))

y_pred = lasso_model.predict(X_test)
print(y_pred)
print(np.sqrt(mean_squared_error(y_test, y_pred)))

lasso_cv_model = LassoCV(alphas = None,
                         cv = 10,
                         max_iter = 10000,
                         normalize = True)

print(lasso_cv_model.fit(X_train,y_train))
print(lasso_cv_model.alpha_)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
print(lasso_tuned.fit(X_train, y_train))
y_pred = lasso_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))

