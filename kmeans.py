# CSC3022F ML Assignment 1
# K-means Clustering
# Author: WNGJIA001

# importing packages
import numpy as np
import array
import copy

# global variables
K = 3
data_points = np.array([[2,10], [2,5], [8,4], [5,8], [7,5], [6,4], [1,2], [4,9]])
centroids = np.array([[2.,10.], [5.,8.], [1.,2.]]) # initial centroids
centroids_prev = np.array([[0.,0.], [0.,0.], [0.,0.]]) # centroids of previous iteration
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
    # make a copy of centroids
    centroids_prev = copy.deepcopy(centroids)
    # make a copy of clusters and clear clusters for updating data-points
    clusters_prev = copy.deepcopy(clusters)
    clusters.clear()
    for i in range(K):
        clusters.append([]);
    # Assign data-points to their closest cluster centroid
    # according to the Euclidean distance function
    for i in range(data_points.shape[0]):
        distance = 10 # distance between data-point and centroid, default = 10
        assign_to = 0 # which cluster the data-point is assigned to, default = 0
        for j in range(K):
            # determine the distance between every data-point and each centroid
            dist = np.linalg.norm(data_points[i]-centroids[j])
            if dist < distance:
                # if the j centroid is closer to the i data-point
                # change the distance and assign_to
                distance = dist
                assign_to = j
        # update which cluster the i data-point belongs to
        clusters[assign_to].append(i+1)
    # Calculate the new centroid or mean of all data-points in each cluster
    for i in range(K):
        i_lst = [] # list to store the data-point in each cluster
        for j in clusters[i]:
            i_lst.append(data_points[j-1])
        # convert the list to numpy array for mean calculation
        i_arr = np.array(i_lst)
        centroids[i] = np.mean(i_arr, axis=0)

if __name__=='__main__':
    i = 1
    while not(convergence()):
        print("Iteration %2d" % (i))
        iteration()
        for j in range(K):
            print("\tCluster %2d:" % (j+1), end = " ")
            s_arr = [str(data_point) for data_point in clusters[j]]
            print(', '.join(s_arr))
            print("\tCentroid: (%.3g, %.3g)" % (centroids[j][0], centroids[j][1]))
            print()
        i += 1
