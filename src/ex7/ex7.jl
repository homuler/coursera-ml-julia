## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.jl
#     projectData.jl
#     recoverData.jl
#     computeCentroids.jl
#     findClosestCentroids.jl
#     kMeansInitCentroids.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
using PyCall

include("findClosestCentroids.jl")
include("computeCentroids.jl")
include("runkMeans.jl")
include("plotProgresskMeans.jl")
include("kMeansInitCentroids.jl")

@pyimport scipy.io as si

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you shoudl complete the code in the findClosestCentroids function.
#
@printf("Finding closest centroids.\n\n")

# Load an example dataset that we will be using
data = si.loadmat("ex7data2.mat")
X = data["X"]

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = [3 3; 6 2; 8 5]

# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids)

@printf("Closest centroids for the first 3 examples: \n")
@printf(" %d %d %d", idx[1], idx[2], idx[3])
@printf("\n(the closest centroids should be 1, 3, 2 respectively)\n")

@printf("Program paused. Press enter to continue.\n")

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
@printf("\nComputing centroids means.\n\n")

# Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

@printf("Centroids computed after initial finding of closest centroids: \n")
display(centroids)
@printf("\n(the centroids should be\n")
@printf("   [ 2.428301 3.157924 ]\n")
@printf("   [ 5.813503 2.633656 ]\n")
@printf("   [ 7.119387 3.616684 ]\n\n")

@printf("Program paused. Press enter to continue.\n")
readline()


## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided.
#
@printf("\nRunning K-Means clustering on example dataset.\n\n")

# Load an example dataset
data = si.loadmat("ex7data2.mat")

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = [3 3; 6 2; 8 5]

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, true)
@printf("\nK-Means Done.\n\n")

@printf("Program paused. Press enter to continue.\n")
readline()

## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel on to it's closest centroid.
#
#  You should now complete the code in kMeansInitCentroids.m
#

@printf("\nRunning K-Means clustering on pixels from an image.\n\n")

#  Load an image of a bird
A = imread("bird_small.png")

# If imread does not work for you, you can try instead
data = si.loadmat("bird_small.mat")

A = A / 255 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = size(A)

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size[1] * img_size[2], 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly.
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

@printf("Program paused. Press enter to continue.\n")
readline()


## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we

@printf("\nApplying K-Means to compress an image.\n\n")

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx.

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value

X_recovered = centroids[idx, :]

# Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size[1], img_size[2], 3)

# Display the original image
subplot(1, 2, 1)
imagesc(A)
title("Original")

# Display compressed image side by side
subplot(1, 2, 2)
imagesc(X_recovered)
title(@sprintf("Compressed, with %d colors.", K))


@printf("Program paused. Press enter to continue.\n")
readline()
