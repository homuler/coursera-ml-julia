using Gadfly

include("multivariateGaussian.jl")

@doc """
  VISUALIZEFIT Visualize the dataset and its estimated distribution.
    VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
""" ->
function visualizeFit(X, mu, sigma2)

  G = [i for j = 0:.5:35, i = 0:.5:35]
  X1 = G[:]
  X2 = G[:]'
  Z = multivariateGaussian([X1[:] X2[:]], mu, sigma2)

  ls = []
  l1 = layer(x=X[:, 1], y=X[:, 2], Geom.point, Theme(default_point_size=2pt))
  push!(ls, l1)

  if size(sigma2, 1) == 1 || size(sigma2, 2) == 1
    sigma2 = diagm(vec(sigma2))
  end

  sinv = pinv(sigma2)

  function Zfunc(a, b)
    x = [a b]
    p = 1 / (2pi * (det(sigma2) ^ (1/2))) * exp(-1/2 * (x - mu) * sinv * (x - mu)')
    return p[1]
  end
  # Do not plot if there are infinities
  if (sum(isinf(Z)) == 0)
    l2 = layer(x=collect(0:.5:35), y=collect(0:.5:35), z=Zfunc,
            Geom.contour(levels=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.05, 0.1]))
    push!(ls, l2)
  end

  return ls
end
