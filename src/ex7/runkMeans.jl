using PyPlot

include("plotProgresskMeans.jl")

@doc """
  RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
  is a single example
    [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    plot_progress) runs the K-Means algorithm on data matrix X, where each
    row of X is a single example. It uses initial_centroids used as the
    initial centroids. max_iters specifies the total number of interactions
    of K-Means to execute. plot_progress is a true/false flag that
    indicates if the function should also plot its progress as the
    learning happens. This is set to false by default. runkMeans returns
    centroids, a Kxn matrix of the computed centroids and idx, a m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
""" ->
function runkMeans(X, initial_centroids,max_iters, plot_progress = false)

  # Plot the data if we are plotting progress
  if plot_progress
    figure()
    hold(true)
  end

  # Initialize values
  m, n = size(X)
  K = size(initial_centroids, 1)
  centroids = initial_centroids
  previous_centroids = centroids
  idx = zeros(m, 1)

  # Run K-Means
  for i = 1:max_iters
    # Output progress
    @printf("K-Means iteration %d/%d...\n", i, max_iters)

    # For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X, centroids);

    # Optionally, plot progress here
    if plot_progress
      #plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
      #previous_centroids = centroids
      #@printf("Press enter to continue.\n")
      #readline()
    end

    # Given the memberships, compute new centroids
    centroids = computeCentroids(X, idx, K)
  end

  # Hold off if we are plotting progress
  if plot_progress
    hold(false)
  end

  return (centroids, idx)
end
