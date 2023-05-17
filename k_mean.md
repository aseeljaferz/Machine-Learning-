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
