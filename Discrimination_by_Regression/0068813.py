import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd


def safelog(x):
    return(np.log(x + 1e-100))


# define the sigmoid function

def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

# define the gradient functions


def gradient_w(X, y_truth, y_predicted):
    return(np.asarray([-np.matmul((y_truth[:, c] - y_predicted[:, c]) * (1 - y_predicted[:, c]) * y_predicted[:, c], X) for c in range(K)]).transpose())


def gradient_w0(y_truth, y_predicted):
    return(-np.sum((y_truth - y_predicted) * (1 - y_predicted) * y_predicted, axis=0))


# importing the data to train and test sets

dataset = np.genfromtxt("hw03_data_set_images.csv", delimiter=",")
labels = np.genfromtxt("hw03_data_set_labels.csv", delimiter=",")


row_start_train = 0
row_end_train = 25
row_start_test = 25
row_end_test = 39
train_dataset = []
test_dataset = []
train_labels = []
test_labels = []

for i in range(5):
    # create training dataset
    a = np.array(dataset[row_start_train:row_end_train, :])
    for j in range(25):
        train_dataset.append(np.array(a[j, :]))

   # create test dataset
    b = np.array(dataset[row_start_test:row_end_test, :])
    for j in range(14):
        test_dataset.append(np.array(b[j, :]))

    # create training dataset labels
    m = labels[row_start_train:row_end_train]
    for j in range(25):
        train_labels.append(np.array(m[j]).astype(int))
    # create test dataset labels
    n = labels[row_start_test:row_end_test]
    for j in range(14):
        test_labels.append(np.array(n[j]).astype(int))
    # update variables to pass the next set of data
    row_start_train = row_start_train + (39)
    row_end_train = row_start_train+(39)
    row_start_test = row_start_test + (39)
    row_end_test = row_end_test+(39)
    
# calculating K and N values
train_dataset = np.array(train_dataset)
train_labels = np.array(train_labels)

test_dataset = np.array(test_dataset)
test_labels = np.array(test_labels)

K = (np.max(train_labels)).astype(int)
N_train = np.array(train_dataset).shape[0]
N_test = np.array(test_dataset).shape[0]


# one-of-K encoding
Y_truth = np.zeros((N_train, K)).astype(int)
Y_truth[range(N_train), train_labels - 1] = 1

# define gradient parameters and point for algorithm to stop

eta = 0.001
epsilon = 0.001

# initalizing w and w0 using random generator function

np.random.seed(521)

w = np.random.uniform(low=-0.01, high=0.01, size=(train_dataset.shape[1], K))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))

#sniprint(w, w0)

# learn w and w0 using gradient descent

iteration = 1
objective_values = []

while True:
    y_pred = sigmoid(train_dataset, w, w0)
    y_pred_test = sigmoid(test_dataset, w, w0)
    
    objective_values = np.append(
        objective_values, 0.5*np.sum((Y_truth - y_pred)**2))

    w_old = w
    w0_old = w0

    w = w - eta * gradient_w(train_dataset, Y_truth, y_pred)
    w0 = w0 - eta * gradient_w0(Y_truth, y_pred)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((w - w_old)**2)) < epsilon:
        break

    iteration = iteration + 1

print(w, w0)

# plot objective/loss function along iterations

plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# calculating confusion matrices

y_pred = np.argmax(y_pred, axis=1) + 1
confusion_matrix = pd.crosstab(y_pred, train_labels, rownames=[
                               'y_pred'], colnames=['y_truth'])
print(confusion_matrix)


y_pred_test = np.argmax(y_pred_test, axis=1) + 1
confusion_matrix = pd.crosstab(y_pred_test, test_labels, rownames=[
                               'y_pred'], colnames=['y_truth'])
print(confusion_matrix)


