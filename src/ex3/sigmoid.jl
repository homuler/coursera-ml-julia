@doc """
  SIGMOID Compute sigmoid functoon
    J = SIGMOID(z) computes the sigmoid of z.
""" ->
function sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z))
  return g
end
