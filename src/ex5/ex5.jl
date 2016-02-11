## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.jl
#     learningCurve.jl
#     validationCurve.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

push!(LOAD_PATH, ".")

using PyCall, DataFrames, Gadfly, Regularization

@pyimport scipy.io as si

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
@printf("Loading and Visualizing Data ...\n")

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = si.loadmat("ex5data1.mat")
X = data["X"]
y = data["y"]
Xval = data["Xval"]
yval = data["yval"]
Xtest = data["Xtest"]
ytest = data["ytest"]

# m = Number of examples
m = size(X, 1)

# Plot training data
p1 = plot(x=X, y=y,Theme(default_point_size=3pt, line_width=1.5pt,
      default_color=colorant"red"), Guide.xlabel("Change in water level (x)"),
      Guide.ylabel("Water flowing out of the dam (y)"))
draw(SVGJS("ex5-dataset.js.svg", 8inch, 6inch), p1)
@printf("Program paused. Press enter to continue.\n")
readline()

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.
#

theta = [1 ; 1]
J, _ = linearRegCostFunction([ones(m, 1) X], y, theta, 1)

@printf("""
  Cost at theta = [1 ; 1]: %f
  (this value should be about 303.993192)
""", J)

@printf("Program paused. Press enter to continue.\n")
readline()

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.
#

theta = [1 ; 1]
J, grad = linearRegCostFunction([ones(m, 1) X], y, theta, 1)

@printf("""
  Gradient at theta = [1 ; 1]:  [%f; %f]
  (this value should be about [-15.303016; 598.250744])
""", grad[1], grad[2])

@printf("Program paused. Press enter to continue.\n")
readline()

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.
#

#  Train linear regression with lambda = 0
lambda = 0
theta = trainLinearReg([ones(m, 1) X], y, lambda)

#  Plot fit over the data
l1 = layer(x=X, y=y, Geom.point,
        Theme(default_point_size=3pt, line_width=1.5pt, default_color=colorant"red"))
l2 = layer(x=X, y=[ones(m, 1) X]*theta, Geom.line, Theme(line_width=2pt))
p2 = plot(l1, l2, Guide.xlabel("Change in water level (x)"), Guide.ylabel("Water flowing out of the dam (y)"))
draw(SVGJS("ex5-linear-regression.js.svg", 8inch, 6inch), p2)

@printf("Program paused. Press enter to continue.\n")
readline()


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
#

lambda = 0
error_train, error_val = learningCurve([ones(m, 1) X], y,
                                       [ones(size(Xval, 1), 1) Xval], yval, lambda)

df1 = DataFrame(x=1:m, y=reshape(error_train, (m,)), label="Train")
df2 = DataFrame(x=1:m, y=reshape(error_val, (m,)), label="Cross Validation")
df = vcat(df1, df2)

p3 = plot(df, x="x", y="y", color="label", Geom.line,
        Guide.title("Learning curve for linear regression"),
        Guide.xlabel("Number of training examples"), Guide.ylabel("Error"),
        Stat.xticks(ticks=collect(0:2:13)), Stat.yticks(ticks=collect(0:50:150)),
        Scale.color_discrete())
draw(SVGJS("ex5-learning-curve.js.svg", 8inch, 6inch), p3)

@printf("# Training Examples\tTrain Error\tCross Validation Error\n")
for i in 1:m
  @printf("  \t%d\t\t%f\t%f\n", i, error_train[i], error_val[i])
end

@printf("Program paused. Press enter to continue.\n")
readline()

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = hcat(ones(m, 1), X_poly)             # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test .- mu
X_poly_test = X_poly_test ./ sigma
X_poly_test = hcat(ones(size(X_poly_test, 1), 1), X_poly_test)    # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val .- mu
X_poly_val = X_poly_val ./ sigma
X_poly_val = hcat(ones(size(X_poly_val, 1), 1), X_poly_val)       # Add Ones

@printf("Normalized Training Example 1:\n")
@printf("%s\n", join(map(x -> @sprintf("  %f  ", x), X_poly[1, :]), ""))

@printf("\nProgram paused. Press enter to continue.\n")
readline()

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lambda = 0
theta = trainLinearReg(X_poly, y, lambda)

# Plot training data and fit
l1 = layer(x=X, y=y, Geom.point, Theme(default_point_size=3pt, default_color=colorant"red"))
l2 = plotFit(minimum(X), maximum(X), mu, sigma, theta, p)
p4 = plot(l1, l2, Guide.title(@sprintf("Polynomial Regression Fit (lambda = %f)", lambda)),
      Guide.xlabel("Change in water level (x)"), Guide.ylabel("Water flowing out of the dam (y)"),
      Stat.xticks(ticks=collect(-80:20:80)), Stat.yticks(ticks=collect(-100:20:200)))
draw(SVGJS("ex5-polynomial-regression.js.svg", 8inch, 6inch), p4)

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda)

df1 = DataFrame(x=1:m, y=reshape(error_train, (m,)), label="Train")
df2 = DataFrame(x=1:m, y=reshape(error_val, (m,)), label="Cross Validation")
df = vcat(df1, df2)

p5 = plot(df, x="x", y="y", color="label", Geom.line,
        Guide.title(@sprintf("Polynomial Regression Learning Curve (lambda = %f)", lambda)),
        Guide.xlabel("Number of training examples"), Guide.ylabel("Error"),
        Stat.xticks(ticks=collect(0:2:13)), Stat.yticks(ticks=collect(0:50:100)),
        Scale.color_discrete())
draw(SVGJS("ex5-learning-curve-reg.js.svg", 8inch, 6inch), p5)

@printf("Polynomial Regression (lambda = %f)\n\n", lambda)
@printf("# Training Examples\tTrain Error\tCross Validation Error\n")
for i in 1:m
    @printf("  \t%d\t\t%f\t%f\n", i, error_train[i], error_val[i])
end

@printf("Program paused. Press enter to continue.\n")
readline()

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

# close all
df1 = DataFrame(x=reshape(lambda_vec, (size(lambda_vec, 1),)),
                y=reshape(error_train, (size(error_train, 1),)), label="Train")
df2 = DataFrame(x=reshape(lambda_vec, (size(lambda_vec, 1),)),
                y=reshape(error_val, (size(error_val, 1),)), label="Cross Validation")
df = vcat(df1, df2)

p6 = plot(df, x="x", y="y", color="label", Geom.line,
        Guide.title(@sprintf("Polynomial Regression Learning Curve (lambda = %f)", lambda)),
        Guide.xlabel("lambda"), Guide.ylabel("Error"),
        Stat.xticks(ticks=collect(0:2:10)), Stat.yticks(ticks=collect(0:5:20)),
        Scale.color_discrete())
draw(SVGJS("ex5-validation-curve-reg.js.svg", 8inch, 6inch), p6)

@printf("lambda\t\tTrain Error\tValidation Error\n")
for i in 1:length(lambda_vec)
	@printf(" %f\t%f\t%f\n", lambda_vec[i], error_train[i], error_val[i])
end

@printf("Program paused. Press enter to continue.\n")
readline()
