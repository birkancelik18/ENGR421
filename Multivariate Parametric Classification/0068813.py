import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import math


def safelog(x):
    return(np.log(x + 1e-100))
# mean parameters
class_means = np.array([[+0, +4.5],
                        [-4.5, -1.00],
                        [+4.5, -1.00],
                        [+0.0, -4.00]])
# covariance parameters
class_covariances = np.array([[[+3.2, +0.0],
                               [+0.0, +1.2]],
                              [[+1.2, +0.8],
                               [+0.8, +1.2]],
                              [[+1.2, -0.8],
                               [-0.8, +1.2]],
                              [[+1.2, +0.0],
                               [+0.0, +3.2]]])
# sample sizes
class_sizes = np.array([105, 145, 135, 115])

# generate random samples
points1 = np.random.multivariate_normal(
    class_means[0, :], class_covariances[0, :, :], class_sizes[0])

points2 = np.random.multivariate_normal(
    class_means[1, :], class_covariances[1, :, :], class_sizes[1])

points3 = np.random.multivariate_normal(
    class_means[2, :], class_covariances[2, :, :], class_sizes[2])

points4 = np.random.multivariate_normal(
    class_means[3, :], class_covariances[3, :, :], class_sizes[3])

X = np.vstack((points1, points2, points3, points4))

# generate corresponding labels CHECK HERE
y = np.concatenate(
    (np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]),
     np.repeat(3, class_sizes[2]), np.repeat(4, class_sizes[3])))
# write data to a file
np.savetxt("hw1_data_set.csv", np.hstack((X, y[:, None])), fmt="%f,%f,%d")

# plot data points generated
plt.figure(figsize=(8, 8))
plt.plot(points1[:, 0],  points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.",  markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.",  markersize=10)
plt.plot(points4[:, 0], points4[:, 1], "m.",  markersize=10)
plt.xlim((-8, +8))
plt.ylim((-8, +8))
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# read data into memory
data_set = np.genfromtxt("hw1_data_set.csv", delimiter=",")

# get X and y values
X = data_set[:, [0, 1]]
y_truth = data_set[:, 2].astype(int)

# get number of samples
K = np.max(y_truth)
N = data_set.shape[0]

# calculate sample means
sample_means = [np.mean(X[y == (c + 1)], axis=0) for c in range(K)]
print("sample means: \n", np.array(sample_means))

# calculate sample covariances
sample_covariances = [(np.matmul(np.transpose(X[y == (c + 1)] - sample_means[c]),
                       (X[y == (c + 1)] - sample_means[c]))) / class_sizes[c] for c in range(K)]

print("sample cov: \n", np.array(sample_covariances))

# calculate prior probabilities
#[105/500, 145/500, 135/500, 115/500]
class_priors = [np.mean(y == (c + 1)) for c in range(K)]

print("class priors \n", class_priors)

W_c = [-0.5 * np.linalg.inv(sample_covariances[c]) for c in range(K)]
w_c = [np.matmul(np.linalg.inv(sample_covariances[c]),
                 sample_means[c]) for c in range(K)]
dimension = X.shape[1]
# print(dim)
w_c0 = [-0.5 * np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covariances[c])), sample_means[c])
        - (dimension/2) * np.log(2*math.pi)
        - (1/2) * np.log(np.linalg.det(sample_covariances[c]))
        + np.log(class_priors[c])
        for c in range(K)]

prediction_y = np.stack([np.array([np.matmul(np.matmul(np.transpose(X[n]), W_c[c]), X[n]) +
                        np.matmul(np.transpose(w_c[c]), X[n]) + w_c0[c] for n in range(N)]) for c in range(K)])

prediction_y = np.argmax(prediction_y, 0) + 1
confusion_matrix = pd.crosstab(prediction_y, y_truth, rownames=[
                               'y_pred'], colnames=['y_truth'])
print(confusion_matrix)


x1_interval = np.linspace(-8, +8, 1601)
x2_interval = np.linspace(-8, +8, 1601)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))


# In[12]:


for c in range(K):
    discriminant_values[:, :, c] = W_c[c][0][0] * x1_grid ** 2 + W_c[c][1][1] * x2_grid ** 2 + \
        W_c[c][1][0] * x1_grid * x2_grid + w_c[c][0] * \
        x1_grid + w_c[c][1] * x2_grid + w_c0[c]

value_1 = discriminant_values[:, :, 0]
value_2 = discriminant_values[:, :, 1]
value_3 = discriminant_values[:, :, 2]
value_4 = discriminant_values[:, :, 3]

value_1[(value_1 < value_2) & (value_1 < value_3)
        & (value_3 < value_4)] = np.nan
value_2[(value_2 < value_1) & (value_2 < value_3)
        & (value_2 < value_4)] = np.nan
value_3[(value_3 < value_1) & (value_3 < value_2)
        & (value_3 < value_4)] = np.nan
value_4[(value_4 < value_1) & (value_4 < value_2)
        & (value_4 < value_3)] = np.nan

discriminant_values[:, :, 0] = value_1
discriminant_values[:, :, 1] = value_2
discriminant_values[:, :, 2] = value_3
discriminant_values[:, :, 3] = value_4

plt.figure(figsize=(10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize=10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize=10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize=10)
plt.plot(X[y_truth == 4, 0], X[y_truth == 4, 1], "m.", markersize=10)
plt.plot(X[prediction_y != y_truth, 0], X[prediction_y != y_truth, 1],
         "ko", markersize=12, fillstyle="none")

"""


plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] -
             discriminant_values[:, :, 1], colors=["g", "r"], levels=0, alpha=0.2)
plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] -
             discriminant_values[:, :, 2], colors=["b", "r"], levels=0, alpha=0.2)
plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] -
             discriminant_values[:, :, 3] -discriminant_values[:,:, 1] -discriminant_values[:,:, 2], colors=["m", "r"], levels=0, alpha=0.2)

plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 1] -
             discriminant_values[:, :, 2], colors=["b", "g"], levels=0, alpha=0.2)
plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 1] -
               discriminant_values[:, :, 3]  , colors=["g", "m"], levels=0, alpha=0.2)
plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 3] - discriminant_values[:, :, 2] 
              , colors=[ "r","m"], levels=0, alpha=0.2)
"""

plt.xlabel("x1")
plt.ylabel("x2")

plt.show()