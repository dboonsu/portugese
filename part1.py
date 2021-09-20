import math
import numpy as np
import pandas as pd
import random
from sklearn.metrics import r2_score # I only used the r^2 calculation from sklearn, for repotring metrics

def _train_test_split(x, y, test_size):
    if test_size >= 1 or test_size <= 0: # Test_size should be a float 0 < x < 1
        exit(-1)
    rows = len(y)
    x_test = np.arange((int)(rows * test_size) * 6).reshape((int)(rows * test_size), 6) # rows * test_size number of rows with 6 columns
    y_test = np.arange((int)(rows * test_size)).reshape((int)(rows * test_size)) # rows * test_size number of rows with 1 column
    # x_test = np.empty((int)(rows * test_size), dtype=object)
    # y_test = np.empty((int)(rows * test_size), dtype=object)
    for i in range((int)(rows * test_size)):
        rand = random.randint(0, len(x) - 1)    # Select a random row
        x_test[i] = x[rand]                     # Insert the random row into test_x
        x = np.delete(x, rand, axis=0)          # Remove the row from training_x
        y_test[i] = y[rand]                     # Same follows for y
        y = np.delete(y, rand)
    return x_test, y_test, x, y

def cost(x, y, m):
    yhat = x.dot(m)
    MSE = (1/rows) * sum((y - yhat) ** 2)
    return MSE

data = pd.read_csv("https://raw.githubusercontent.com/dboonsu/portugese/master/student-mat.csv", delimiter=";") #portuguese kid's math scores
data.replace(('yes', 'no'), (1, 0), inplace=True) # Very light preprocessing
data.replace(('U', 'R'), (1, 0), inplace=True)
# 13 - studytime, 15 - school support, 16 - family support, 30 - Grades 1, 31 - Grades 2
cols = [13, 15, 16, 30, 31] # Uses these columns to predict Grades 3
x = data.values[:, cols]
colNames = []
colNames.append('y-intercept')
columns = np.shape(x)[1] + 1 # plus one is for the 1's column
for i in range(columns - 1):
    colNames.append(data.columns[cols[i]])
y = data.values[:, 32] # G3 - Grades 3, predicted value

learning_rate = 0.008
convergence = .00005
iterations = 25000
MSE = 0;
MSE_history = []

rows = len(y)
x = np.insert(x, 0, np.ones(rows), axis=1)
m = np.zeros(columns)

x_test, y_test, x, y = _train_test_split(x, y, .2)

for iter in range(iterations):
    yhat = x.dot(m) # Predicted values
    errors = np.subtract(yhat, y) # Calculate errors
    m = m - learning_rate * (1 / rows) * x.transpose().dot(errors); # alpha * 1/n * sum(x * (yhat - y))
    MSE_ = cost(x, y, m)
    MSE_history.append(MSE_)
    if (abs(MSE_ - MSE) <= convergence):
        print("Convergence limit hit (" + str(iter) + " iterations)")
        break;
    MSE = MSE_

# Output to console
for i in range(columns):
    print(str(colNames[i]) + ": " + str(m[i]))
MSE_train = MSE
print("\nTraining Data: ")
r2_train = r2_score(y, yhat)
print("MSE: " + str(MSE_train))
print("R2: " + str(r2_train))

print("\nTesting Data: ")
yhat = x_test.dot(m) # Predicted values
errors = np.subtract(yhat, y_test) # Calculate errors
MSE_test = cost(x_test, y_test, m)
r2_test = r2_score(y_test, yhat)
print("MSE: " + str(MSE_test))
print("R2: " + str(r2_test))
# End output to console

# Output to file
logFile = open("log.txt", "a")
logFile.write("\n----------------------------------------")
logFile.write("\nLearning Rate: " + str(learning_rate))
logFile.write("\nMax Iterations: "  + str(iterations))
logFile.write("\nIterations Taken: " + str(iter))
logFile.write("\nConvergence Rate: " + str(convergence))
logFile.write("\nMSE Test: " + str(MSE_test))
logFile.write("\nr2 Test: " + str(r2_test))
logFile.write("\nMSE Training: " + str(MSE_train))
logFile.write("\nr2 Training: " + str(r2_train))
logFile.write("\nWeight Coefficients")
for i in range(columns):
    logFile.write("\n" + str(colNames[i]) + ": " + str(m[i]))
# End output to file

