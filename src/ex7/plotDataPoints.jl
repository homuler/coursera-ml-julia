using Gadfly, DataFrames

@doc """
  PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
  index assignments in idx have the same color
    PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
    with the same index assignments in idx have the same color
""" ->
function plotDataPoints(X, idx, K)

  colors = Scale.color_discrete().f(K)
  # Plot the data
  df = convert(DataFrame, [X idx])

  ls = []

  for sg in groupby(df, :x3)
    l = layer(sg, x="x1", y="x2", Theme(default_color=colors[convert(Int32, sg[:x3][1])]), Geom.point)
    push!(ls, l)
  end
  return ls
end
