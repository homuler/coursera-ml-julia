@doc """
  MAPFEATURE Feature mapping function to polynomial features

    MAPFEATURE(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size
""" ->
function mapFeature(X1, X2)
  degree = 6
  out = ones(size(X1[:, 1]))
  for i in 1:degree, j in 0:i
    out = hcat(out, (X1 .^ (i-j)) .* (X2 .^ j))
  end
  return out
end
