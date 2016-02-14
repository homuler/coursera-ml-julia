@doc """
  KMEANSINITCENTROIDS This function initializes K centroids that are to be
  used in K-Means on the dataset X
    centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    used with the K-Means on the dataset X
""" ->
function kMeansInitCentroids(X, K)

  # You should return this values correctly
  centroids = zeros(K, size(X, 2))

  randidx = randperm(size(X, 1))
  centroids = X[randidx[1:K], :]
  # ====================== YOUR CODE HERE ======================
  # Instructions: You should set centroids to randomly chosen examples from
  #               the dataset X
  #
  # =============================================================
  return centroids
end
