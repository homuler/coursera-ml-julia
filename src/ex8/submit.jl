export submit

include("../data.jl")
include("../submit.jl")

push!(LOAD_PATH, ".")

using AnomalyDetection
using CollaborativeFiltering: cofiCostFunc

function submit()
  parts = [
    Part(1, "Estimate Gaussian Parameters"),
    Part(2, "Select Threshold"),
    Part(3, "Collaborative Filtering Cost"),
    Part(4, "Collaborative Filtering Gradient"),
    Part(5, "Regularized Cost"),
    Part(6, "Regularized Gradient")
  ]
  conf = Conf("anomaly-detection-and-recommender-systems",
              "Anomaly Detection and Recommender Systems", parts, solver)

  submitWithConf(conf)
end

function solver(partId)
  # Random Test Cases
  n_u = 3
  n_m = 4
  n = 5

  X = reshape(sin(1:n_m*n), n_m, n)
  Theta = reshape(cos(1:n_u*n), n_u, n)
  Y = reshape(sin(1:2:2*n_m*n_u), n_m, n_u)
  R = Y .> 0.5
  pval = [abs(Y[:]) ; 0.001; 1]
  yval = [R[:] ; 1; 0]
  params = [X[:]; Theta[:]]

  if partId == 1
    mu, sigma2 = estimateGaussian(X)
    return join(map(x -> @sprintf("%0.5f", x), [mu[:]; sigma2[:]]), " ")
  elseif partId == 2
    bestEpsilon, bestF1 = selectThreshold(yval, pval)
     return join(map(x -> @sprintf("%0.5f", x), [bestEpsilon; bestF1]), " ")
  elseif partId == 3
    J, grad = cofiCostFunc(params, Y, R, n_u, n_m, n, 0)
    return @sprintf("%0.5f", J)
  elseif partId == 4
    J, grad = cofiCostFunc(params, Y, R, n_u, n_m, n, 0)
    return join(map(x -> @sprintf("%0.5f", x), grad[:]), " ")
  elseif partId == 5
    J, grad = cofiCostFunc(params, Y, R, n_u, n_m, n, 1.5)
    return @sprintf("%0.5f", J)
  elseif partId == 6
    J, grad = cofiCostFunc(params, Y, R, n_u, n_m, n, 1.5)
    return join(map(x -> @sprintf("%0.5f", x), grad[:]), " ")
  end
end
