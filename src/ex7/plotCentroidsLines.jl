using Gadfly, DataFrames

function plotCentroidsLines(centroids, previous, K)

  ls = []

  for i in 1:size(centroids, 1)
    ps = [centroids[i, :]; previous[i]]
    l = layer(x=ps[:, 1], y=ps[:, 2], Theme(default_color=colorant"black"), Geom.point, Geom.line)
    push!(ls, l)
  end
  return ls
end
