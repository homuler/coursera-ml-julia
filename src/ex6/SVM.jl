module SVM

include("model.jl")
include("svmTrain.jl")
include("svmPredict.jl")
include("linearKernel.jl")
include("gaussianKernel.jl")
include("dataset3Params.jl")

export SVMModel, svmTrain, svmPredict, linearKernel, gaussianKernel, dataset3Params

end
