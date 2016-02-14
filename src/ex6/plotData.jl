using Gadfly

@doc """
  PLOTDATA Plots the data points X and y into a new figure
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    Note: This was slightly modified such that it expects y = 1 or y = 0
""" ->
function plotData(X, y)
  # Find Indices of Positive and Negative Examples
  pos = find(y .== 1)
  neg = find(y .== 0)

  # Plot Examples
  ls = []
  l1 = layer(x=X[pos, 1], y=X[pos, 2], Geom.point, Theme(default_point_size=3pt))
  l2 = layer(x=X[neg, 1], y=X[neg, 2], Geom.point, Theme(default_point_size=3pt, default_color=colorant"yellow"))

  push!(ls, l1)
  push!(ls, l2)
  return ls
end
