module MultiClassClassification

include("lrCostFunction.jl")
include("oneVsAll.jl")
include("predictOneVsAll.jl")
include("predict.jl")

export lrCostFunction, oneVsAll, predictOneVsAll, predict

end
