export submit

include("../data.jl")
include("../submit.jl")

include("linearRegCostFunction.jl")
include("learningCurve.jl")
include("polyFeatures.jl")
include("validationCurve.jl")

function submit()
  parts = [
    Part(1, "Regularized Linear Regression Cost Function"),
    Part(2, "Regularized Linear Regression Gradient"),
    Part(3, "Learning Curve"),
    Part(4, "Polynomial Feature Mapping"),
    Part(5, "Validation Curve")
  ]
  conf = Conf("regularized-linear-regression-and-bias-variance",
              "Regularized Linear Regression and Bias/Variance", parts, solver)

  submitWithConf(conf)
end

function solver(partId)
  # Random Test Cases
  X = [ones(10,1) sin(1:1.5:15) cos(1:1.5:15)]
  y = sin(1:3:30)
  Xval = [ones(10,1) sin(0:1.5:14) cos(0:1.5:14)]
  yval = sin(1:10)
  if partId == 1
    J, _ = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5)
    return @sprintf("%0.5f", J)
  elseif partId == 2
    J, grad = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5)
    return join(map(x -> @sprintf("%0.5f", x), grad), " ")
  elseif partId == 3
    error_train, error_val = learningCurve(X, y, Xval, yval, 1)
    return join(map(x -> @sprintf("%0.5f", x), [error_train[:]; error_val[:]]), " ")
  elseif partId == 4
    X_poly = polyFeatures(X[2, :], 8)
    return join(map(x -> @sprintf("%0.5f", x), X_poly), " ")
  elseif partId == 5
    lambda_vec, error_train, error_val = validationCurve(X, y, Xval, yval)
    return join(map(x -> @sprintf("%0.5f", x), [lambda_vec[:]; error_train[:]; error_val[:]]), " ")
  end
end
