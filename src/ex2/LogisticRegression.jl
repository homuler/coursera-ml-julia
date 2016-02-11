module LogisticRegression

include("plotData.jl")
include("costFunction.jl")
include("costFunctionReg.jl")
include("plotDecisionBoundary.jl")
include("sigmoid.jl")
include("predict.jl")
include("mapFeature.jl")

export plotData, costFunction, costFunctionReg, plotDecisionBoundary,
       sigmoid, predict, mapFeature

end
