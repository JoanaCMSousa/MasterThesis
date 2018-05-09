# Importing the libraries
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
path = '/Users/Joana/Documents/Churn_Modelling.json'
dataset = pd.read_json(path)


c = [x for x in range(14) if x != 5 and x != 3 and x != 11 and x != 12]
X = dataset.iloc[:, c].values
y = dataset.iloc[:, 5].values

labelencoder_X_1 = LabelEncoder()
X[:, 5] = labelencoder_X_1.fit_transform(X[:, 5])
labelencoder_X_2 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])

onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting our model
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#estimator = KerasClassifier(build_fn= classifier, epochs= 100, batch_size= 10)

#################
#kfold = StratifiedKFold(n_splits= 10, shuffle=True, random_state= seed)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#results = cross_val_score(estimator, X_train, y_train)

print(cm)