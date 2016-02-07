module AnomalyDetection

include("estimateGaussian.jl")
include("multivariateGaussian.jl")
include("visualizeFit.jl")
include("selectThreshold.jl")

export estimateGaussian, multivariateGaussian, visualizeFit, selectThreshold

end
