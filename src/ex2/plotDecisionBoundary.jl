include("plotData.jl")

@doc """
  PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
  the decision boundary defined by theta
    PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    positive examples and o for the negative examples. X is assumed to be
    a either
    1) Mx3 matrix, where the first column is an all-ones column for the
        intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
""" ->
function plotDecisionBoundary(theta, X, y)
  # Plot Data
  plotData(X[:, 2:3], y)
  hold(true)

  if size(X, 2) <= 3
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = [minimum(X[:, 2])-2,  maximum(X[:, 2])+2]

    # Calculate the decision boundary line
    plot_y = (-1./theta[3]).*(theta[2].*plot_x + theta[1])

    # Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)

    # Legend, specific for the exercise
    # legend("Admitted", "Not admitted", "Decision Boundary")
    legend("Admitted", "Not admitted")
    axis([30, 100, 30, 100])
  else
    # Here is the grid range
    u = linspace(-1, 1.5, 50)
    v = linspace(-1, 1.5, 50)

    z = zeros(length(u), length(v))
    # Evaluate z = theta*x over the grid
    for j in 1:length(v), i in 1:length(u)
      z[i, j] = (mapFeature([u[i]], [v[j]]) * theta)[1]
    end
    z = z' # important to transpose z before calling contour

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], linewidth=2)
  end
  hold(false)
end
