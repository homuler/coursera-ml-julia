include("computeCost.jl")

@doc """
  GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
""" ->
function gradientDescent(X, y, theta, alpha, num_iters)
  # Initialize some useful values
  m = length(y)

  J_history = zeros(num_iters)
  for iter in 1:num_iters
    # ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector
    #               theta.
    #
    # Hint: While debugging, it can be useful to print out the values
    #       of the cost function (computeCost) and gradient here.
    # ============================================================
    # Save the cost J in every iteration
    J_history[iter] = computeCost(X, y, theta)
  end
  return (theta, J_history)
end
