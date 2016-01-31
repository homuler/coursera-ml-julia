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
using PyCall, PyPlot

@pyimport scipy.io as si

include("featureNormalize.jl")
include("pca.jl")
include("projectData.jl")
include("recoverData.jl")
include("displayData.jl")
include("kMeansInitCentroids.jl")
include("runkMeans.jl")

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
@printf("Visualizing example dataset for PCA.\n\n")

#  The following command loads the dataset. You should now have the
#  variable X in your environment
data = si.loadmat("ex7data1.mat")
X = data["X"]

#  Visualize the example dataset
#plot(X[:, 1], X[:, 2], marker='o', color='b')
#axis([0.5 6.5 2 8])
#axis("equal")

@printf("Program paused. Press enter to continue.\n")
readline()


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
@printf("\nRunning PCA on example dataset.\n\n")

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.

#hold(true)
#drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
#drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
#hold(false)

@printf("Top eigenvector: \n")
@printf(" U(:,1) = %f %f \n", U[1, 1], U[2, 1])
@printf("\n(you should expect to see -0.707107 -0.707107)\n")

@printf("Program paused. Press enter to continue.\n")
readline()


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the
#  first k eigenvectors. The code will then plot the data in this reduced
#  dimensional space.  This will show you what the data looks like when
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
@printf("\nDimension reduction on example dataset.\n\n")

#  Plot the normalized dataset (returned from pca)
#plot(X[:, 1], X[:, 2], marker='o', color='b')
#axis([-4 3 -4 3])
#axis("equal")

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
@printf("Projection of the first example: %f\n", Z[1])
@printf("\n(this value should be about 1.481274)\n\n")

X_rec  = recoverData(Z, U, K)
@printf("Approximation of the first example: %f %f\n", X_rec[1, 1], X_rec[1, 2])
@printf("\n(this value should be about  -1.047419 -1.047419)\n\n")

#  Draw lines connecting the projected points to the original points
#hold(true)
#plot(X_rec[:, 1], X_rec[:, 2], marker='o', color='r')
#for i = 1:size(X_norm, 1)
#    drawLine(X_norm[i, :], X_rec[i, :], '--k', 'LineWidth', 1)
#end
#hold(false)

@printf("Program paused. Press enter to continue.\n")
readline()

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
@printf("\nLoading face dataset.\n\n")

#  Load Face dataset
data = si.loadmat("ex7faces.mat")

#  Display the first 100 faces in the dataset
# displayData(X[1:100, :])

@printf("Program paused. Press enter to continue.\n")
readline()

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
@printf("""\nRunning PCA on face dataset.
  (this mght take a minute or two ...)
""")

#  Before running PCA, it is important to first normalize X by subtracting
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Visualize the top 36 eigenvectors found
# displayData(U[:, 1:36]')

@printf("Program paused. Press enter to continue.\n")
readline()


## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
@printf("\nDimension reduction for face dataset.\n\n")

K = 100
Z = projectData(X_norm, U, K)

@printf("The projected data Z has a size of: ")
@printf("%d %d", size(Z, 1), size(Z, 2))

@printf("\n\nProgram paused. Press enter to continue.\n")
readline()

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

@printf("\nVisualizing the projected (reduced dimension) faces.\n\n")

K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
#subplot(1, 2, 1)
#displayData(X_norm[1:100, :])
#title("Original faces")
#axis("equals")

# Display reconstructed data from only k eigenfaces
#subplot(1, 2, 2)
#displayData(X_rec[1:100, :])
#title("Recovered faces")
#axis("equals")

@printf("Program paused. Press enter to continue.\n")
readline()


## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = imread("bird_small.png")

# If imread does not work for you, you can try instead
#   load ('bird_small.mat');

A = A / 255
img_size = size(A)
X = reshape(A, img_size[1] * img_size[2], 3)
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = rand(1000, 1) * size(X, 1) + 1

#  Setup Color Palette
#palette = hsv(K)
#colors = palette(idx[sel], :)

#  Visualize the data and centroid memberships in 3D
#figure;
#scatter3(X[sel, 1], X[sel, 2], X[sel, 3], 10, colors);
#title("Pixel dataset plotted in 3D. Color shows centroid memberships")
@printf("Program paused. Press enter to continue.\n");
readline()

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
#figure()
#plotDataPoints(Z[sel, :], idx[sel], K)
#title("Pixel dataset plotted in 2D, using PCA for dimensionality reduction")
@printf("Program paused. Press enter to continue.\n");
readline()
