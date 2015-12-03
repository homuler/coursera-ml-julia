using PyPlot

include("polyFeatures.jl")

@doc """
  PLOTFIT Plots a learned polynomial regression fit over an existing figure.
  Also works with linear regression.
    PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    fit with power p and feature normalization (mu, sigma).
""" ->
function plotFit(min_x, max_x, mu, sigma, theta, p)
  # Hold on to the current figure
  hold(true)

  # We plot a range slightly bigger than the min and max values to get
  # an idea of how the fit will vary outside the range of the data points
  x = (min_x - 15:0.05:max_x + 25)

  # Map the X values
  X_poly = polyFeatures(x, p)
  X_poly = X_poly .- mu
  X_poly = X_poly ./ sigma

  # Add ones
  X_poly = [ones(size(x, 1), 1) X_poly]

  println(size(x), " ", size(X_poly * theta))
  # Plot
  plot(x, X_poly * theta, linestyle="--", linewidth=2, color="b")

  # Hold off to the current figure
  hold(false)
end
