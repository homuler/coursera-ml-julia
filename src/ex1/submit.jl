export submit

include("../data.jl")
include("../submit.jl")

include("warmUpExercise.jl")
include("computeCost.jl")
include("gradientDescent.jl")
include("featureNormalize.jl")
include("computeCostMulti.jl")
include("gradientDescentMulti.jl")
include("normalEqn.jl")

function submit()
  parts = [
    Part(1, "Warm-up Exercise"),
    Part(2, "Computing Cost (for One Variable)"),
    Part(3, "Gradient Descent (for One Variable)"),
    Part(4, "Feature Normalization"),
    Part(5, "Computing Cost (for Multiple Variables)"),
    Part(6, "Gradient Descent (for Multiple Variables)"),
    Part(7, "Normal Equations")
  ]
  conf = Conf("linear-regression", "Linear Regression with Multiple Variables", parts, solver)

  submitWithConf(conf);
end

function solver(partId)
  # Random Test Cases
  X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))]
  Y1 = X1[:, 2] + sin(X1[:, 1]) + cos(X1[:, 2])
  X2 = [X1 X1[:, 2].^0.5 X1[:, 2].^0.25]
  Y2 = Y1.^0.5 + Y1
  if partId == 1
    return join(map(x -> @sprintf("%0.5f ", x), warmUpExercise()), " ")
  elseif partId == 2
    return @sprintf("%0.5f ", computeCost(X1, Y1, [0.5 -0.5]'))
  elseif partId == 3
    return join(map(x -> @sprintf("%0.5f ", x), gradientDescent(X1, Y1, [0.5 -0.5]', 0.01, 10)[1:size(X1)[2]]), " ")
  elseif partId == 4
    return join(map(x -> @sprintf("%0.5f ", x), featureNormalize(X2[:, 2:4]))[1:size(X2)[1], :], " ")
  elseif partId == 5
    return @sprintf("%0.5f ", computeCostMulti(X2, Y2, [0.1 0.2 0.3 0.4]'));
  elseif partId == 6
    return join(map(x -> @sprintf("%0.5f ", x), gradientDescentMulti(X2, Y2, [-0.1 -0.2 -0.3 -0.4]', 0.01, 10)[1:size(X2)[2]]), " ");
  elseif partId == 7
    return join(map(x -> @sprintf("%0.5f ", x), normalEqn(X2, Y2)), " ");
  end
end
