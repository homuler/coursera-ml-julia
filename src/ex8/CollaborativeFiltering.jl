module CollaborativeFiltering

include("estimateGaussian.jl")
include("cofiCostFunc.jl")
include("selectThreshold.jl")
include("checkCostFunction.jl")
include("computeNumericalGradient.jl")
include("loadMovieList.jl")
include("normalizeRatings.jl")

export estimateGaussian, cofiCostFunc, selectThreshold,
       checkCostFunction, computeNumericalGradient, loadMovieList,
       normalizeRatings

end
