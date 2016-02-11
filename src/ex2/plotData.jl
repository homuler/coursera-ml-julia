using Gadfly, DataFrames

@doc """
  PLOTDATA Plots the data points X and y into a new figure
    PLOTDATA(x,y) returns the layer of Gadfly, which plots the data points with blue color
    for the positive examples and yellow for the negative examples. X is assumed to be a Mx2 matrix.
""" ->
function plotData(X, y)

  # ====================== YOUR CODE HERE ======================
  # Instructions: Plot the positive and negative examples on a
  #               2D plot, using the option 'k+' for the positive
  #               examples and 'ko' for the negative examples.
  #
  # =========================================================================
  ls = []
  l1 = layer(x=X[:, 1], y=X[:, 2], Geom.point)
  push!(ls, l1)
  return ls
end
