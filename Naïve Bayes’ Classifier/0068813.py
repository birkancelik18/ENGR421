import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()

# precaiton in case of log(0) returns inf


def safelog(x):
    return(np.log(x + 1e-100))


# importing the data to train and test sets

dataset = np.genfromtxt("hw02_data_set_images.csv", delimiter=",")
labels = np.genfromtxt("hw02_data_set_labels.csv", delimiter=",")


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


# mean parameters (pcd)

training_mean_values = np.array(
    [np.mean(train_dataset[train_labels == (c + 1)], axis=0) for c in range(K)])
print(training_mean_values.shape)


# deviations

training_deviation_values = np.array([np.sqrt(np.mean(
    (train_dataset[train_labels == c + 1] - training_mean_values[c]) ** 2, axis=0)) for c in range(K)])
# print(np.array(training_deviation_values))

# class priors

class_priors = [
    np.mean(np.array(train_labels).astype(int) == (m+1)) for m in range(K)]

print(class_priors)


# score values for the training set

training_scores = []

for i in range(train_dataset.shape[0]):
    training_scores.append([np.sum(train_dataset[i] * safelog(training_mean_values[c]) + (
        1-train_dataset[i]) * safelog(1-training_mean_values[c])) + safelog(class_priors[c]) for c in range(K)])
training_scores = np.array(training_scores)



pred_y = np.argmax(training_scores, 1) + 1
confusion_matrix = pd.crosstab(pred_y, train_labels, rownames=[
                               'y_pred'], colnames=['y_truth'])
print(confusion_matrix)


# score values for the test set
test_scores = []

for i in range(test_dataset.shape[0]):
    test_scores.append([np.sum(test_dataset[i] * safelog(training_mean_values[c]) +
                               (1-test_dataset[i]) * safelog(1-training_mean_values[c])) + safelog(class_priors[c]) for c in range(K)])
test_scores = np.array(test_scores)
print(test_scores.shape)


pred_y = np.argmax(test_scores, 1) + 1
confusion_matrix = pd.crosstab(pred_y, test_labels, rownames=[
                               'y_pred'], colnames=['y_truth'])
print(confusion_matrix)
