# CSC3022F ML Assignment 1
# K-means Clustering
# Author: WNGJIA001

# importing packages
import numpy as np
import array

# global variables
data_points = np.array([[2,10], [2,5], [8,4], [5,8], [7,5], [6,4], [1,2], [4,9]])
centroids = np.array([[2,10], [5,8], [1,2]]) # initial centroids
centroids_prev = np.array([[0,0], [0,0], [0,0]]) # centroids of previous iteration
clusters = [[], [], []] # initial clusters K = 3
clusters_prev = [[], [], []] # clusters of previous iteration

def convergence():
    # convergence criterion:
    # 1. no re-assignments of data-points to different clusters
    # 2. no change of centroids
    # 3. minimum decrease in SSE (assumed to be 0 for this assignment)
    if np.array_equal(centroids,centroids_prev) and (clusters == clusters_prev):
        return True
    return False

def iteration():
    global centroids, centroids_prev, clusters, clusters_prev
    centroids_prev = centroids
    clusters_prev = clusters
    # Assign data-points to their closest cluster centroid
    # according to the Euclidean distance function
    for i in range(data_points.shape[0]):
        distance = 10 # distance between data-point and centroid, default = 10
        assign_to = 0 # which cluster the data-point is assigned to, default = 0
        for j in range(centroids.shape[0]):
            # determine the distance between every data-point and each centroid
            dist = np.linalg.norm(data_points[i]-centroids[j])
            # print("distance of data-point ", i, "to centroid ", j, ": ", dist)
            if dist < distance:
                # if the j centroid is closer to the i data-point
                # change the distance and assign_to
                distance = dist
                assign_to = j
        # print("assign data point", i, "to cluster", assign_to)
        clusters[assign_to].append(i)
    print(clusters)

    # Calculate the centroid or mean of all objects in each cluster

if __name__=='__main__':
    iteration()
