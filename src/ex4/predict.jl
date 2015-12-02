@doc """
  PREDICT Predict the label of an input given a trained neural network
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
""" ->
function predict(Theta1, Theta2, X)
  # Useful values
  m = size(X, 1)
  num_labels = size(Theta2, 1)

  # You need to return the following variables correctly
  p = zeros(size(X, 1), 1)

  h1 = sigmoid([ones(m, 1) X] * Theta1')
  h2 = sigmoid([ones(m, 1) h1] * Theta2')

  for i in 1:m
    p[i] = findmax(h2[i, :])[2]
  end
  return p
end
