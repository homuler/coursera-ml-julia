using PyPlot, PyCall

@pyimport numpy as np
@pyimport matplotlib.pyplot as pyplot

include("plotData.jl")
include("svmPredict.jl")

@doc """
  VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    boundary learned by the SVM and overlays the data on it
""" ->
function visualizeBoundary(X, y, model)

  # Plot the training data on top of the boundary
  plotData(X, y)

  # Make classification predictions over a grid of values
  x1plot = linspace(minimum(X[: ,1]), maximum(X[:, 1]), 100)'
  x2plot = linspace(minimum(X[:, 2]), maximum(X[:, 2]), 100)'
  X1, X2 = np.meshgrid(x1plot, x2plot)
  vals = zeros(size(X1))
  for i = 1:size(X1, 2)
    this_X = [X1[:, i] X2[:, i]]
    vals[:, i] = svmPredict(model, this_X)
  end

  # Plot the SVM boundary
  hold(true)
  println(size(X1), size(X2), size(vals))
  pyplot.contour(X1, X2, vals, levels=[0 0], color="b")
  hold(false)
end
