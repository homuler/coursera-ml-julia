@doc """
  FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
""" ->
function featureNormalize(X)

  mu = mean(X)
  X_norm = X .- mu

  sigma = std(X_norm)
  X_norm = X_norm ./ sigma

  return (X_norm, mu, sigma)
end
