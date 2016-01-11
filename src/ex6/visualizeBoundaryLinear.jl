using PyPlot

include("plotData.jl")

@doc """
  %VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
  SVM
    VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    learned by the SVM and overlays the data on it
""" ->
function visualizeBoundaryLinear(X, y, model)
  w = model.w
  b = model.b
  xp = linspace(minimum(X[:, 1]), maximum(X[:, 1]), 100)
  yp = - (w[1]*xp + b)/w[2]
  plotData(X, y)
  hold(true)
  plot(xp, yp, "-b")
  hold(false)
end
