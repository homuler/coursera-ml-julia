module CollaborativeFiltering

include("estimateGaussian.jl")
include("cofiCostFunc.jl")
include("selectThreshold.jl")
include("checkCostFunction.jl")

export estimateGaussian, cofiCostFunc, selectThreshold, checkCostFunction

end
