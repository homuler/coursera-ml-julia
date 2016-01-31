module kMeans

include("runkMeans.jl")
include("findClosestCentroids.jl")
include("computeCentroids.jl")
include("kMeansInitCentroids.jl")

export runkMeans, findClosestCentroids, computeCentroids, kMeansInitCentroids

end
