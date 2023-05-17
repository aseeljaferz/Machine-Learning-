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
