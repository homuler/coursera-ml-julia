## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     warmUpExercise.jl
#     plotData.jl
#     gradientDescent.jl
#     computeCost.jl
#     gradientDescentMulti.jl
#     computeCostMulti.jl
#     featureNormalize.jl
#     normalEqn.jl
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization
push!(LOAD_PATH, ".")

using PyPlot: surf, xlabel, ylabel, ColorMap, @L_str
using Gadfly, LinearRegression

## ================ Part 1: Feature Normalization ================

## Clear and Close Figures

@printf("Loading data ...\n")

## Load Data
data = readcsv("ex1data2.txt")
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

# Print out some data points
@printf("First 10 examples from the dataset: \n")
println(join([@sprintf("x = [%.0f %0.f], y = %.0f", X[i, 1], X[i, 2], y[i, 1]) for i in 1:10], "\n"))

@printf("Program paused. Press enter to continue.\n")
readline()

# Scale features and set them to zero mean
@printf("Normalizing Features ...\n")
X, mu, sigma = featureNormalize(X)
# Add intercept term to X
X = [ones(m, 1) X]


## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

@printf("Running gradient descent ...\n")

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent
theta = zeros(3, 1)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
p1 = plot(x=collect(1:length(J_history)), y=J_history, Geom.smooth(method=:loess, smoothing=0.9),
      Theme(line_width=2pt, default_color=colorant"blue"), Guide.xlabel("Number of iterations"), Guide.ylabel("Cost J"))
draw(SVGJS("ex1-linear-regression-multi.js.svg", 6inch, 6inch), p1)

# Display gradient descent's result
println("Theta computed from gradient descent:")
println(join(map(x -> @sprintf("%f", x), theta), " "))

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = 0 # You should change this


# ============================================================

@printf("""
  Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
   \$%f
""", price)

println("Program paused. Press enter to continue.")
readline()

## ================ Part 3: Normal Equations ================

println("Solving with normal equations...")

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load Data
data = readcsv("ex1data2.txt")
X = data[:, 1:2]
y = data[:, 3]
m = length(y)

# Add intercept term to X
X = [ones(m, 1) X]

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
println("Theta computed from the normal equations:")
println(join(map(x -> @sprintf("%f", x), theta), " "))

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = 0 # You should change this


# ============================================================

@printf("""
  Predicted price of a 1650 sq-ft, 3 br house (using normal equations):
   \$%f
""", price)
