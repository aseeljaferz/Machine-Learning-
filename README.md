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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3    naive bayes classification:

#load the iris ataset
from sklearn.datasets import load_iris
iris = load_iris()

#store the feature matrix (x) and response vector (y)
X = iris.data
Y = iris.target 

#splitting x and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

#training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)

#making predictions on the testing set
Y_pred = gnb.predict(X_test)

#comparing actual response values (Y_test) with predicted response values (Y_pred) 
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy (in %)",metrics.accuracy_score(Y_test, Y_pred)*100)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4    using decision tree:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from google.colab import files
uploaded=files.upload()

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes (1).csv", header=0, names=col_names)
pima.head()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] 
y = pima.label 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy in testing set:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred_test = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_test))

from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

5      classification using support vector machine:

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("diabetes1.csv")
data.head()

non_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for coloumn in non_zero:
    data[coloumn] = data[coloumn].replace(0,np.NaN)
    mean = int(data[coloumn].mean(skipna = True))
    data[coloumn] = data[coloumn].replace(np.NaN,mean)
    print(data[coloumn])
    
from sklearn.model_selection import train_test_split
X =data.iloc[:,0:8]
y =data.iloc[:,8]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0, stratify=y)
X.head() 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn import svm
from sklearn.svm import SVC
svc = SVC()
svm1 = svm.SVC(kernel="linear", C = 0.01)
svm1.fit(X_test,y_test)
SVC(C=0.01, kernel="linear")
y_train_pred = svm1.predict(X_train)
y_test_pred = svm1.predict(X_test)
y_test_pred

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_test_pred)

accuracy_score(y_test,y_test_pred)

---------------------------------------------------------------------------------------------------------------

6    multivariate classification/regression:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

uploaded = files.upload()


uploaded = files.upload()



test = pd.read_csv("california_housing_test.csv")
# test.head()




train = pd.read_csv("california_housing_train.csv")
# train.head()




plt.figure()
sns.heatmap(train.corr(), cmap = 'coolwarm')
plt.show()
sns.lmplot(x = 'median_income', y = 'median_house_value', data = train)
sns.lmplot(x = 'housing_median_age', y = 'median_house_value', data = train)




data = train
data = data[['total_rooms', 'total_bedrooms', 'housing_median_age' ,'median_income', 'population', 'households']] 
data.info()
data['total_rooms'] = data['total_rooms'].fillna(data['total_rooms'].mean())
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())




from sklearn.model_selection import train_test_split
y = train.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 0)




print(y.name)




from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)




print(regressor.intercept_)
print(regressor.coef_)




predictions = regressor.predict(X_test)
predictions = predictions.reshape(-1,1)
print(predictions)




from sklearn.metrics import mean_squared_error
print('A:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))


--------------------------------------------------------------------------------------------------------------

7    feed forward neureal network:



import math
import pandas as pd 
from keras import models, layers, optimizers, regularizers
import numpy as np
import random
from sklearn import model_selection, preprocessing
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt




from google.colab import files

uploaded = files.upload()




data = pd.read_csv("SAheart.data", sep=',', index_col=0)
data['famhist'] = data['famhist'] == 'Present'
data.head()




n_test = int(math.ceil(len(data) * 0.3))
random.seed(42)
test_ixs = random.sample(list(range(len(data))), n_test)
train_ixs = [ix for ix in range(len(data)) if ix not in test_ixs]
train = data.iloc[train_ixs, :]
test = data.iloc[test_ixs, :]
print(len(train))
print(len(test))




#features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']
features = ['adiposity', 'age']
response = 'chd'
x_train = train[features]
y_train = train[response]
x_test = test[features]
y_test = test[response]




x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)




hidden_units = 10     # how many neurons in the hidden layer
activation = 'relu'   # activation function for hidden layer
l2 = 0.01             # regularization - how much we penalize large parameter values
learning_rate = 0.01  # how big our steps are in gradient descent
epochs = 5            # how many epochs to train for
batch_size = 16       # how many samples to use for each gradient descent update




# create a sequential model
model = models.Sequential()

# add the hidden layer
model.add(layers.Dense(input_dim=len(features),
                       units=hidden_units, 
                       activation=activation))

# add the output layer
model.add(layers.Dense(input_dim=hidden_units,
                       units=1,
                       activation='sigmoid'))

# define our loss function and optimizer
model.compile(loss='binary_crossentropy',
              # Adam is a kind of gradient descent
              optimizer=optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])




# train the parameters
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size)

# evaluate accuracy
train_acc = model.evaluate(x_train, y_train, batch_size=32)[1]
test_acc = model.evaluate(x_test, y_test, batch_size=32)[1]
print('Training accuracy: %s' % train_acc)
print('Testing accuracy: %s' % test_acc)

losses = history.history['loss']
plt.plot(range(len(losses)), losses, 'r')
plt.show()

### RUN IT AGAIN! ###

-----------------------------------------------------------------------------------------------------------------------

8      k-mean clustering:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as metrics




from google.colab import files 
upload = files.upload() 




df = pd.read_csv('user.csv')
df




x = df.iloc[:,[0,1]].values
print(x)




kmeans2 = KMeans(n_clusters=2)
y_kmeans2 = kmeans2.fit_predict(x)
print(y_kmeans2)
print("Cluster centers are:")
print(kmeans2.cluster_centers_)




plt.scatter(x[:,0],x[:,1],c=y_kmeans2,cmap='viridis')
plt.show()

-----------------------------------------------------------------------------------------

10         cross validation:



import numpy as np




from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




from sklearn import datasets




from sklearn import svm




X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape




X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=0)




X_train.shape, y_train.shape




X_test.shape, y_test.shape




clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)




Y_predict = clf.predict(X_test)
print(classification_report(Y_predict,y_test))




from sklearn.model_selection import cross_val_score




clf = svm.SVC(kernel='linear', C=1, random_state=42)




scores = cross_val_score(clf, X, y, cv=10)




scores

