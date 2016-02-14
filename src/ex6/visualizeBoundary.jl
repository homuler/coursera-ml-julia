using Gadfly

include("plotData.jl")
include("svmPredict.jl")

@doc """
  VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
    boundary learned by the SVM and overlays the data on it
""" ->
function visualizeBoundary(X, y, model)

  # Plot the training data on top of the boundary
  ls = plotData(X, y)

  # Make classification predictions over a grid of values
  x1plot = linspace(minimum(X[: ,1]), maximum(X[:, 1]), 100)'
  x2plot = linspace(minimum(X[:, 2]), maximum(X[:, 2]), 100)'

  vals = zeros(100, 100)

  for i = 1:100
    this_X = [[x1plot[i] for j in 1:100] x2plot']
    vals[i, :] = svmPredict(model, this_X)
  end

  aIdxMap = [x1plot[i] => i for i in 1:100]
  bIdxMap = [x2plot[i] => i for i in 1:100]

  function Zfunc(a, b)
    # this_X = [[a for i in 1:100] x2plot']
    # vals = svmPredict(model, this_X)
    a_idx = get(aIdxMap, a, 0)
    b_idx = get(bIdxMap, b, 0)
    return vals[a_idx, b_idx]
  end
  # Plot the SVM boundary
  l = layer(x=x1plot, y=x2plot, z=Zfunc, Geom.contour(levels=[0]), Theme(default_color=colorant"blue"))
  push!(ls, l)
  return ls
end
