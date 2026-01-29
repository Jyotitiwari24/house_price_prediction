import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from xgboost import XGBRegressor
from sklearn import metrics

from sklearn.datasets import fetch_california_housing
house_price_dataset = fetch_california_housing()

print(house_price_dataset)

# Loading the Dataset to a Pandas Dataset
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)

# Print First 5 Rows of Dataset
house_price_dataframe.head()

# Add the Target Price Column to the Dataframe
house_price_dataframe['price']= house_price_dataset.target

house_price_dataframe.head()

# Checking the number of rows and columns in the data frame
house_price_dataframe.shape

# check for missing values
house_price_dataframe.isnull().sum()

# statistical measures of the dataset
house_price_dataframe.describe()

correlation = house_price_dataframe.corr()

from pickle import TRUE
# constructing a heatmap to understand the correlation
plt.figure(figsize= (10,10))
sns.heatmap(correlation, cbar= True , square= True, fmt= '1f' , annot= True , annot_kws={'size' : 8}, cmap = 'Blues')

X= house_price_dataframe.drop(['price'], axis = 1)
Y = house_price_dataframe['price']

print(X)
print(Y)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , random_state= 2)

print(X.shape ,X_test.shape, X_train.shape)

print(Y.shape, Y_train.shape, Y_test.shape)

# Loading the Model
model = XGBRegressor()

# Training the Model with X_train
model.fit(X_train, Y_train)

# Accuracy for Prediction on Training Data
training_data_prediction = model.predict(X_train)

print(training_data_prediction)

#  R Squared Erroe
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Square Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print('R Squared Error :' , score_1)
print('Mean Absolute Error :' , score_2)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel('Actual Prices :')
plt.ylabel('Predicted Prices :')
plt.title('Actual Prices VS Predicted Prices. ')
plt.show()

# Accuracy for Prediction on Test data
test_data_prediction = model.predict(X_test)

print(training_data_prediction)



#  R Squared Erroe
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Square Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print('R Squared Error :' , score_1)
print('Mean Absolute Error :' , score_2)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel('Actual Prices :')
plt.ylabel('Predicted Prices :')
plt.title('Actual Prices VS Predicted Prices. ')
plt.show()
