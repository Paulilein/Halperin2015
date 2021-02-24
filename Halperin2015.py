# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Halperin2015_SI_Data.csv', sep=';')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
# Want y to be in a 2D array [[]]. right now, its a vector
y = y.reshape(len(y),1)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

# Training the SVR model on the training dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train.ravel())

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
y_pred = y_pred.reshape(len(y_pred), 1)

y_test = sc.inverse_transform(y_test)
y_error = abs(y_test-y_pred)
y_normalized_error = y_error/y_test
accuracy = (1-np.mean(y_normalized_error))*100
print(accuracy)

# Visualising the SVR results
plt.scatter(np.linspace(1,len(y_test),len(y_test)), y_test, color = 'red', label = 'real values')
plt.scatter(np.linspace(1,len(y_test),len(y_test)), y_pred, color = 'blue', label = 'predicted values')
for i in range(len(y_error)):
    label = None
    if i == 1:
        label = 'diff'
    plt.plot(np.array([i+1,i+1]), np.array([y_test[i], y_pred[i]]), color = 'black', label = label)
plt.legend()
plt.title('Halperin2015 (Support Vector Regression)')
plt.xticks(ticks=[i+1 for i in range(len(y_test))])
plt.ylabel('Area []')
plt.show()