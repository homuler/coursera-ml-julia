using NLopt

include("linearRegCostFunction.jl")

@doc """
  TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
  regularization parameter lambda
    [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    the dataset (X, y) and regularization parameter lambda. Returns the
    trained parameters theta.
""" ->
function trainLinearReg(X, y, lambda)
  # Initialize Theta
  initial_theta = zeros(size(X, 2))

  # Create "short hand" for the cost function to be minimized
  function costFunction(theta, grad)
    cost, gs = linearRegCostFunction(X, y, theta, lambda)
    if length(grad) > 0
      grad[:] = gs
    end
    return cost
  end

  # Now, costFunction is a function that takes in only one argument
  options = Opt(:LD_TNEWTON, length(initial_theta))
  maxeval!(options, 200)

  # Minimize using fmincg
  # theta = fmincg(costFunction, initial_theta, options);
  min_objective!(options, costFunction)
  (_, theta, _) = optimize(options, initial_theta)

  return theta
end
