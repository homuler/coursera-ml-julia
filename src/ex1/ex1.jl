# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
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
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
push!(LOAD_PATH, ".")

using PyPlot: surf, xlabel, ylabel, ColorMap, @L_str
using Gadfly, LinearRegression

include("warmUpExercise.jl")


## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m

@printf "Running warmUpExercise ... \n"
@printf "5x5 Identity Matrix: \n"
println(warmUpExercise())

@printf "Program paused. Press enter to continue.\n"
readline()

## ======================= Part 2: Plotting =======================
@printf("Plotting Data ...\n")
data = readcsv("ex1data1.txt")
X = data[:, 1]
y = data[:, 2]
m = length(y) # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m

l1 = plotData(X, y)
draw(SVGJS("ex1-dataset.js.svg", 6inch, 6inch), plot(l1))

@printf "Program paused. Press enter to continue.\n"
readline()

## =================== Part 3: Gradient descent ===================
@printf "Running Gradient Descent ...\n"

X = [ones(m) data[:, 1]] # Add a column of ones to x
theta = zeros(2, 1) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
println(computeCost(X, y, theta))

# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)
theta = theta[1:2]

# print theta to screen
@printf "Theta found by gradient descent: "
@printf "%f %f \n" theta[1] theta[2]

# Plot the linear fit
l2 = layer(x=X[:, 2], y=X*theta, Geom.line, Theme(default_color=colorant"red"))
draw(SVGJS("ex1-linear-regression.js.svg", 6inch, 6inch), plot(l1, l2))

# Predict values for population sizes of 35,000 and 70,000
predict1 = [1 3.5] * theta
@printf "For population = 35,000, we predict a profit of %f\n" (predict1*10000)[1]
predict2 = [1.0 7] * theta
@printf "For population = 70,000, we predict a profit of %f\n" (predict2*10000)[1]

@printf "Program paused. Press enter to continue.\n"

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
@printf "Visualizing J(theta_0, theta_1) ...\n"

# Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals))

# Fill out J_vals
for i in 1:length(theta0_vals), j in 1:length(theta1_vals)
	t = [theta0_vals[i]; theta1_vals[j]]
	J_vals[i, j] = computeCost(X, y, t)
end

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals'
# Surface plot
surf(theta0_vals, theta1_vals, J_vals, rstride=5, cstride=5, cmap=ColorMap("rainbow"))
xlabel(L"\theta_0")
ylabel(L"\theta_1")

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
function Zfunc(a, b)
	println(computeCost(X, y, [a; b]))
	return computeCost(X, y, [a; b])
end

l1 = layer(x=collect(theta0_vals), y=collect(theta1_vals), z=Zfunc,
				Geom.contour(levels=[1e3, 2e3, 3e3, 5e3, 7e3, 1e4, 2e4, 3e4, 5e4, 7e4, 1e5, 5e5, 1e6]))
l2 = layer(x=[theta[1]], y=[theta[2]], Geom.point, Theme(default_point_size=3pt))
draw(SVGJS("ex1-contour.js.svg", 7inch, 7inch), plot(l1, l2, Guide.xlabel("θ0"), Guide.ylabel("θ1")))
