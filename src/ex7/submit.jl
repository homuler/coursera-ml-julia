export submit

include("../data.jl")
include("../submit.jl")

include("findClosestCentroids.jl")
include("computeCentroids.jl")
include("pca.jl")
include("projectData.jl")
include("recoverData.jl")

function submit()
  parts = [
    Part(1, "Find Closest Centroids (k-Means)"),
    Part(2, "Compute Centroid Means (k-Means)"),
    Part(3, "PCA"),
    Part(4, "Project Data (PCA)"),
    Part(5, "Recover Data (PCA)")
  ]
  conf = Conf("k-means-clustering-and-pca",
              "K-Means Clustering and PCA", parts, solver)

  submitWithConf(conf)
end

function solver(partId)
  # Random Test Cases
  X = reshape(sin(1:165), 15, 11)
  Z = reshape(cos(1:121), 11, 11)
  C = Z[1:5, :]
  idx = (1 + mod(1:15, 3))'

  if partId == 1
    idx = findClosestCentroids(X, C)
    return join(map(x -> @sprintf("%0.5f", x), idx[:]), " ")
  elseif partId == 2
    centroids = computeCentroids(X, idx, 3)
    return join(map(x -> @sprintf("%0.5f", x), centroids[:]), " ")
  elseif partId == 3
    U, S = pca(X)
    return join(map(x -> @sprintf("%0.5f", x), abs([U[:]; S[:]])), " ")
  elseif partId == 4
    X_proj = projectData(X, Z, 5)
    return join(map(x -> @sprintf("%0.5f", x), X_proj[:]), " ")
  elseif partId == 5
    X_rec = recoverData(X[:, 1:5], Z, 5)
    return join(map(x -> @sprintf("%0.5f", x), X_rec[:]), " ")
  end
end
