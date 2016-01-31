using PyPlot

@doc """
  PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
  index assignments in idx have the same color
    PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
    with the same index assignments in idx have the same color
""" ->
function plotDataPoints(X, idx, K)

  # Create palette
  display(hsv)
  palette = get_cmap("hsv", K + 1)
  # colors = palette[idx, :]

  # Plot the data
  scatter(X[:, 1], X[:, 2], 15, cmap=palette)
end
