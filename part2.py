import pandas as pd
import numpy as np
import random

data = pd.read_csv("https://raw.githubusercontent.com/dboonsu/portugese/master/student-mat.csv", delimiter=";") #portuguese kid's math scores
data.replace(('yes', 'no'), (1, 0), inplace=True)
data.replace(('U', 'R'), (1, 0), inplace=True)
cols = [13, 15, 16, 30, 31] # Uses these columns to predict Grades 3
X = data.values[:, cols]
Y = data.values[:, 32]
colNames = []
columns = np.shape(X)[1] + 1 # plus one is for the 1's column
for i in range(columns - 1):
    colNames.append(data.columns[cols[i]])

learning_rate = 0.001
convergence = .0001
iterations = 25000

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

lin_model = SGDRegressor(alpha = learning_rate, max_iter = iterations, tol=convergence)
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
MSE_train = mean_squared_error(Y_train, y_train_predict)
r2_train = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(MSE_train))
print('R2 score is {}'.format(r2_train))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
MSE_test = mean_squared_error(Y_test, y_test_predict)
r2_test = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(MSE_test))
print('R2 score is {}'.format(r2_test))
print("\n\n\n")

logFile = open("log.txt", "a")
logFile.write("\n----------------------------------------")
logFile.write("\nLearning Rate: " + str(learning_rate))
logFile.write("\nMax Iterations: "  + str(iterations))
logFile.write("\nIterations Taken: " + str(lin_model.n_iter_))
logFile.write("\nConvergence Rate: " + str(convergence))
logFile.write("\nMSE Test: " + str(MSE_test))
logFile.write("\nr2 Test: " + str(r2_test))
logFile.write("\nMSE Training: " + str(MSE_train))
logFile.write("\nr2 Training: " + str(r2_train))
logFile.write("\nWeight Coefficients")
logFile.write("\ny-intercept: " + str(lin_model.intercept_[0]))
for i in range(columns - 1):
    logFile.write("\n" + str(colNames[i]) + ": " + str(lin_model.coef_[i]))
# End output to file