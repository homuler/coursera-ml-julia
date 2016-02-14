@doc """
  LINEARKERNEL returns a linear kernel between x1 and x2
    sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
    and returns the value in sim
""" ->
function linearKernel(x1, x2)
  # Ensure that x1 and x2 are column vectors
  x1 = x1[:]
  x2 = x2[:]

  # Compute the kernel
  sim = vecdot(x1, x2)  # dot product

  return sim
end
