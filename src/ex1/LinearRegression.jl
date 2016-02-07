module LinearRegression

include("plotData.jl")
include("gradientDescent.jl")
include("computeCost.jl")
include("gradientDescentMulti.jl")
include("computeCostMulti.jl")
include("featureNormalize.jl")
include("normalEqn.jl")

export plotData, gradientDescent, computeCost, gradientDescentMulti,
       computeCostMulti, featureNormalize, normalEqn

end
