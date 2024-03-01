import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()

print(digits)

print(digits.DESCR)

print(digits.data)

print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()

model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2, 5, 1 + i)
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap = plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.14,2.11,0.96,0.00,0.00,0.00,0.00,2.63,6.68,7.60,6.11,0.00,0.00,0.00,0.58,7.61,6.71,5.06,7.62,1.29,0.00,0.00,0.00,1.82,0.56,2.33,7.62,1.52,0.00,0.00,1.49,2.28,1.35,5.20,7.54,0.68,0.00,2.03,7.61,7.61,7.62,7.61,6.67,2.35,0.74,0.82,5.84,6.07,5.13,5.99,7.60,7.62,3.96],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.66,3.80,5.16,5.33,5.10,3.04,0.08,0.00,7.61,7.61,7.53,6.47,6.84,7.62,4.65,0.00,7.61,2.73,0.00,0.00,0.00,5.24,6.85,0.00,7.61,3.27,0.00,0.00,0.00,4.10,6.85,0.00,6.85,5.86,0.23,0.00,0.53,6.69,6.01,0.00,3.04,7.62,6.84,5.17,7.08,7.61,2.74,0.00,0.00,2.12,5.40,6.09,5.25,2.34,0.00,0.00],
[0.00,0.00,0.83,1.52,1.51,0.08,0.00,0.00,0.79,5.56,7.62,7.62,7.62,6.76,2.11,0.00,3.96,7.38,4.09,1.66,2.84,6.45,7.38,0.79,0.43,1.00,0.00,0.00,0.00,1.89,7.61,2.27,0.00,0.00,0.00,0.00,0.00,1.88,7.61,2.27,0.91,3.04,3.04,2.26,1.64,5.96,7.60,4.30,4.94,7.61,7.61,7.61,7.61,7.61,7.61,7.54,3.29,7.61,7.60,7.60,6.37,2.72,2.12,0.37],
[0.00,0.00,0.30,1.52,1.52,1.52,0.30,0.00,0.00,0.30,5.84,7.61,7.61,7.61,3.51,0.00,0.00,3.26,7.62,3.80,1.51,1.52,0.30,0.00,0.00,5.25,6.69,0.00,0.00,0.00,0.00,0.00,0.00,5.33,6.23,4.17,6.85,6.54,2.03,0.00,0.00,5.25,7.61,7.62,5.62,7.60,5.24,0.00,0.00,3.04,7.62,7.31,5.92,7.61,3.11,0.00,0.00,0.07,3.16,6.08,6.09,3.72,0.00,0.00]
])

new_labels= model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
