using PyPlot

include("plotDataPoints.jl")

@doc """
  PLOTPROGRESSKMEANS is a helper function that displays the progress of
  k-Means as it is running. It is intended for use only with 2D data.
    PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    points with colors assigned to each centroid. With the previous
    centroids, it also plots a line between the previous locations and
    current locations of the centroids.
""" ->
function plotProgresskMeans(X, centroids, previous, idx, K, i)

  # Plot the examples
  plotDataPoints(X, idx, K)

  # Plot the centroids as black x's
  scatter(centroids[:, 1], centroids[:, 2], marker='x', edgecolor='k', s=10, linewidth=3)
  # Plot the history of the centroids with lines
  for j = 1:size(centroids,1)
    drawLine(centroids[j, :], previous[j, :])
  end

  # Title
  title(@sprintf("Iteration number %d", i))

end
