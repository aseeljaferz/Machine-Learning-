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
