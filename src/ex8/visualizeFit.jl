using Gadfly

include("multivariateGaussian.jl")

@doc """
  VISUALIZEFIT Visualize the dataset and its estimated distribution.
    VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
""" ->
function visualizeFit(X, mu, sigma2)

  X1, X2 = meshgrid(0:.5:35)
  Z = multivariateGaussian([X1[:] X2[:]], mu, sigma2)
  Z = reshape(Z, size(X1))

  plot(x=X[:, 1], y=X[:, 2], Grid.point)

  # Do not plot if there are infinities
  if (sum(isinf(Z)) == 0)
    plot(x=X1, y=X2, z=Z, Grid.contour)
  end

end
