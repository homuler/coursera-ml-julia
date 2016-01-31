using Gadfly

include("plotDataPoints.jl")
include("plotCentroidsLines.jl")

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
  pointGroups = plotDataPoints(X, idx, K)
  centroidGroups = plotCentroidsLines(centroids, previous, K)

  # Plot the centroids as black x's

  draw(SVGJS("ex7-progress-kmeans.js.svg", 10inch, 6inch),
       plot(pointGroups..., centroidGroups..., Guide.title(@sprintf("Iteration number %d", i))))
  # Plot the history of the centroids with lines
  #for j = 1:size(centroids,1)
  #  drawLine(centroids[j, :], previous[j, :])
  #end

end
