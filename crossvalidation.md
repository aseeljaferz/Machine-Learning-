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
