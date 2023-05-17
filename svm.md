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
