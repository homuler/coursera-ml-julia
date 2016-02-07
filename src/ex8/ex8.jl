## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.jl
#     selectThreshold.jl
#     cofiCostFunc.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
push!(LOAD_PATH, ".")

using PyCall, Gadfly, AnomalyDetection

@pyimport scipy.io as si

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#

@printf("Visualizing example dataset for outlier detection.\n\n")

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment
data = si.loadmat("ex8data1.mat")
X = data["X"]
Xval = data["Xval"]
yval = data["yval"]

#  Visualize the example dataset
l1 = layer(x=X[:, 1], y=X[:, 2], Geom.point)

p1 = plot(l1, Guide.xlabel("Latency (ms)"), Guide.ylabel("Throughput (mb/s)"),
  Stat.xticks(ticks=collect(0:5:30)), Stat.yticks(ticks=collect(0:5:30)))

draw(SVGJS("ex8-dataset.js.svg", 10inch, 6inch), p1)

@printf("Program paused. Press enter to continue.\n")
readline()


## ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution,
#  then compute the probabilities for each of the points and then visualize
#  both the overall distribution and where each of the points falls in
#  terms of that distribution.
#
@printf("Visualizing Gaussian fit.\n\n")

#  Estimate my and sigma2
mu, sigma2 = estimateGaussian(X)

#  Returns the density of the multivariate normal at each data point (row)
#  of X
p = multivariateGaussian(X, mu, sigma2)

#  Visualize the fit
ls = visualizeFit(X,  mu, sigma2)
p2 = plot(ls..., Guide.xlabel("Latency (ms)"), Guide.ylabel("Throughput (mb/s)"))
draw(SVGJS("ex8-anomaly-detection.js.svg", 10inch, 6inch), p2)

@printf("Program paused. Press enter to continue.\n")
readline()

## ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set
#  probabilities given the estimated Gaussian distribution
#

pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)
@printf("Best epsilon found using cross-validation: %e\n", epsilon)
@printf("Best F1 on Cross Validation Set:  %f\n", F1)
@printf("   (you should see a value epsilon of about 8.99e-05)\n\n")

#  Find the outliers in the training set and plot the
outliers = find(p .< epsilon)

#  Draw a red circle around those outliers
l3 = layer(x=X[outliers, 1], y=X[outliers, 2], Geom.point, Theme(default_color=colorant"red", default_point_size=4pt))
push!(ls, l3)
p3 = plot(ls..., Guide.xlabel("Latency (ms)"), Guide.ylabel("Throughput (mb/s)"))
draw(SVGJS("ex8-anomaly-detection.js.svg", 10inch, 6inch), p3)

@printf("Program paused. Press enter to continue.\n")
readline()

## ================== Part 4: Multidimensional Outliers ===================
#  We will now use the code from the previous part and apply it to a
#  harder problem in which more features describe each datapoint and only
#  some features indicate whether a point is an outlier.
#

#  Loads the second dataset. You should now have the
#  variables X, Xval, yval in your environment
data = si.loadmat("ex8data2.mat")
X = data["X"]
Xval = data["Xval"]
yval = data["yval"]

#  Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

#  Training set
p = multivariateGaussian(X, mu, sigma2)

#  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

@printf("Best epsilon found using cross-validation: %e\n", epsilon)
@printf("Best F1 on Cross Validation Set:  %f\n", F1)
@printf("# Outliers found: %d\n", sum(p .< epsilon))
@printf("   (you should see a value epsilon of about 1.38e-18)\n\n")
readline()
