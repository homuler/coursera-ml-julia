export submit

include("../data.jl")
include("../submit.jl")

include("sigmoid.jl")
include("costFunction.jl")
include("predict.jl")
include("costFunctionReg.jl")

function submit()
  parts = [
    Part(1, "Sigmoid Function"),
    Part(2, "Logistic Regression Cost"),
    Part(3, "Logistic Regression Gradient"),
    Part(4, "Predict"),
    Part(5, "Regularized Logistic Regression Cost"),
    Part(6, "Regularized Logistic Regression Gradient")
  ]
  conf = Conf("logistic-regression", "Logistic Regression", parts, solver)

  submitWithConf(conf);
end

function solver(partId)
  # Random Test Cases
  X = [ones(20,1) (exp(1) * sin(1:1:20)) (exp(0.5) * cos(1:1:20))]
  y = sin(X[:, 1] + X[:, 2]) .> 0
  if partId == 1
    return join(map(x -> @sprintf("%0.5f ", x), sigmoid(X)), " ")
  elseif partId == 2
    return @sprintf("%0.5f ", costFunction([0.25 0.5 -0.5]', X, y)[1])
  elseif partId == 3
    return join(map(x -> @sprintf("%0.5f", x), costFunction([0.25 0.5 -0.5]', X, y)[2:end]), " ")
  elseif partId == 4
    return join(map(x -> @sprintf("%0.5f", x), predict([0.25 0.5 -0.5]', X)), " ")
  elseif partId == 5
    return @sprintf("%0.5f ", costFunctionReg([0.25 0.5 -0.5]', X, y, 0.1)[1])
  elseif partId == 6
    return join(map(x -> @sprintf("%0.5f ", x), costFunctionReg([0.25 0.5 -0.5]', X, y, 0.1)), " ")
  end
end
