import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical


# type your name-surname and student ID
# read dataset
df = pd.read_csv('breast-cancer.csv')


# do some data preprocessing
# remove unnecessary columns
# convert categorical variables to numerical if nesessary




# split the data into features and target variable
X =
y =


# make categorical the target variable
num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)



# split the data into training and testing sets


# Scale data if you want


# Build ANN model
model = Sequential([


])


# Compile the model


# Train the model

# Test set evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
