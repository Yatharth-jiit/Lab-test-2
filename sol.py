import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score
data = load_diabetes()
print("Number of features:", len(data.feature_names))
x = data.data
y = data.target
plt.plot(x,y)
plt.show()
plt.scatter(x_train,y_train)
plt.show()
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.4, random_state=42)
KNeighborsClassifier()
knn60 = KNeighborsClassifier()
knn60.fit(x_train, y_train)
knn_60 = knn60.predict(x_test)
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
knn80 = KNeighborsClassifier()
knn80.fit(x_train, y_train)
knn_80 = knn80.predict(x_test)
