module Regularization

include("linearRegCostFunction.jl")
include("trainLinearReg.jl")
include("learningCurve.jl")
include("polyFeatures.jl")
include("featureNormalize.jl")
include("plotFit.jl")
include("validationCurve.jl")

export linearRegCostFunction, trainLinearReg, learningCurve,
       polyFeatures, featureNormalize, plotFit, validationCurve
end
