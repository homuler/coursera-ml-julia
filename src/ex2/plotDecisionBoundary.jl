using Gadfly

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
  ls = plotData(X[:, 2:3], y)

  if size(X, 2) <= 3
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = [minimum(X[:, 2])-2,  maximum(X[:, 2])+2]

    # Calculate the decision boundary line
    plot_y = (-1./theta[3]).*(theta[2].*plot_x + theta[1])

    # Plot, and adjust axes for better viewing
    l2 = layer(x=plot_x, y=plot_y, Geom.line)
    push!(ls, l2)
  else
    # Here is the grid range
    u = linspace(-1, 1.5, 50)
    v = linspace(-1, 1.5, 50)

    function Zfunc(a, b)
      return (mapFeature([b], [a]) * theta)[1]
    end

    # Plot z = 0
    l2 = layer(x=u, y=v, z=Zfunc, Theme(line_width=2pt), Geom.contour(levels=[0]))
    push!(ls, l2)
  end

  return ls
end
