1  missing value:
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io
df = pd.read_csv(io.BytesIO(uploaded['StudentsPerformance.csv']))
print(df)
df.fillna({"writing score":df["writing score"].mean()},inplace = True)
print(df)
f.fillna({"reading score":df["reading score"].median()},inplace = True)
print(df)
df["gender"].fillna(method="bfill",inplace = True)
print(df)
df.replace("female","male",inplace=True)print(df)

2  feature selection/extraction to perform dimensionality reduction:

from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest   # sk => sciKit
from sklearn.feature_selection import f_classif
from google.colab import files
uploaded = files.upload()

filename = 'pima-indians-diabetes.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

#------------------------univerified method----------------------------

from sklearn.feature_selection import SelectKBest   # sk => sciKit
from sklearn.feature_selection import f_classif

# feature extraction

test = SelectKBest(score_func=f_classif,k=4)
fit = test.fit(x, y)

# summarize scores

set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(x)

# summarize selected features

print(features[0:5])

#-----------------Recursive Feature Elimination------------------------
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction

model = LogisticRegression(solver = 'lbfgs')
rfe = RFE(model)fit = rfe.fit(x, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

#---------------Principal Component Analysis-------------------

import numpy
from sklearn.decomposition import PCA
# feature extraction

pca = PCA(n_components=3)
fit = pca.fit(X)

# summarize components

print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
