using Gadfly

@doc """
  PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    population and profit.
""" ->
function plotData(x, y)
  # ====================== YOUR CODE HERE ======================

  l1 = layer(x=x, y=y, Geom.point)
  # ============================================================
  return l1
end
