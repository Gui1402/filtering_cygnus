import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

## Using scikit-learn to generate clusters

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1])

## find distance between two closest neighboohrs
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
## as n_neighbors is equal 2, the distances will be 0 and the closest point from that point
distances, indices = nbrs.kneighbors(X)

## now we will sort points and plot distances
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)

## The optimal value for epsilon will be found at the point of maximum curvature.

# Using knee locator to find the optimal point
from kneed import KneeLocator
kn = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
eps_value = distances[kn.knee]

# create a dbscan object
m = DBSCAN(eps=eps_value, min_samples=5, metric='l2')
m.fit(X)

clusters = m.labels_
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
