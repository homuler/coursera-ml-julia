## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.jl
#     costFunction.jl
#     predict.jl
#     costFunctionReg.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.


## Initialization
using PyPlot
using NLopt

include("plotData.jl")
include("costFunction.jl")
include("plotDecisionBoundary.jl")
include("sigmoid.jl")
include("predict.jl")
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = readcsv("ex2data1.txt")
X = data[:, 1:2]
y = data[:, 3]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

@printf("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n")

plotData(X, y)

# Put some labels
hold(true)
# Labels and Legend
xlabel("Exam 1 score")
ylabel("Exam 2 score")

# Specified in plot order
legend(["Admitted", "Not admitted"])
hold(false)

@printf("Program paused. Press enter to continue.\n")
readline()


## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = size(X)

# Add intercept term to x and X_test
X = [ones(m, 1) X]

# Initialize fitting parameters
initial_theta = zeros(n + 1)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

@printf("Cost at initial theta (zeros): %f\n", cost)
@printf("Gradient at initial theta (zeros): \n")
@printf("%s", join(map(x -> @sprintf(" %f ", x), grad), "\n"))

@printf("\nProgram paused. Press enter to continue.\n")
readline()


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
#options = optimset("GradObj", "on", "MaxIter", 400)
options = Opt(:LN_NELDERMEAD, n+1)
min_objective!(options, (theta, grad) -> costFunction(theta, X, y)[1])
#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost
(cost, theta, _) = optimize(options, initial_theta)

# Print theta to screen
@printf("Cost at theta found by optimize: %f\n", cost)
@printf("theta: \n")
@printf("%s", join(map(x -> @sprintf(" %f ", x), theta), "\n"))

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Put some labels
hold(true)
# Labels and Legend
xlabel("Exam 1 score")
ylabel("Exam 2 score")

# Specified in plot order
legend(["Admitted", "Not admitted"])
hold(false)

@printf("\nProgram paused. Press enter to continue.\n")
#readline()

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid([1 45 85] * theta)
@printf("For a student with scores 45 and 85, we predict an admission probability of %f\n\n", prob[1])

# Compute accuracy on our training set
p = predict(theta, X)

@printf("Train Accuracy: %f\n", mean(p .== y) * 100)
