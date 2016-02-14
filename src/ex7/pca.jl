@doc """
  PCA Run principal component analysis on the dataset X
   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
""" ->
function pca(X)

  # Useful values
  m, n = size(X)

  # You need to return the following variables correctly.
  U = zeros(n)
  S = zeros(n)

  cov = X' * X ./ m
  F = svdfact(cov)

  # ====================== YOUR CODE HERE ======================
  # Instructions: You should first compute the covariance matrix. Then, you
  #               should use the "svd" function to compute the eigenvectors
  #               and eigenvalues of the covariance matrix.
  #
  # Note: When computing the covariance matrix, remember to divide by m (the
  #       number of examples).
  #
  # =========================================================================
  return F[:U], diagm(F[:S])
end
