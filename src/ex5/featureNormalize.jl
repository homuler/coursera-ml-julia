@doc """
  FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
""" ->
function featureNormalize(X)
  es = size(X, 2)
  mu = [mean(X[:, i]) for i in 1:es]'
  X_norm = X .- mu

  sigma = [std(X_norm[:, i]) for i in 1:es]'
  X_norm ./= sigma

  return (X_norm, mu, sigma)
end
