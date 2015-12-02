## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoidGradient.jl
#     randInitializeWeights.jl
#     nnCostFunction.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

using PyCall, NLopt

@pyimport scipy.io as si

include("displayData.jl")
include("nnCostFunction.jl")
include("randInitializeWeights.jl")
include("checkNNGradients.jl")
include("predict.jl")
# include("fmincg.jl")

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
@printf("Loading and Visualizing Data ...\n")

data = si.loadmat("ex4data1.mat")
X = data["X"]
y = data["y"]
m = size(X, 1)

# Randomly select 100 data points to display
sel = randperm(size(X, 1))
sel = sel[1:100]

displayData(X[sel, :])

@printf("Program paused. Press enter to continue.\n")
readline()


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

@printf("\nLoading Saved Neural Network Parameters ...\n")

# Load the weights into variables Theta1 and Theta2
nn = si.loadmat("ex4weights.mat")
Theta1 = nn["Theta1"]
Theta2 = nn["Theta2"]

# Unroll parameters
nn_params = [Theta1[:] ; Theta2[:]]

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
@printf("\nFeedforward Using Neural Network ...\n")

# Weight regularization parameter (we set this to 0 here).
lambda = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda)

println(J)
@printf("Cost at parameters (loaded from ex4weights): %f\n(this value should be about 0.287629)\n", J)

@printf("\nProgram paused. Press enter to continue.\n")
readline()

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

@printf("\nChecking Cost Function (w/ Regularization) ... \n")

# Weight regularization parameter (we set this to 1 here).
lambda = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda)

@printf("Cost at parameters (loaded from ex4weights): %f\n(this value should be about 0.383770)\n", J)

@printf("Program paused. Press enter to continue.\n")
readline()


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

@printf("\nEvaluating sigmoid gradient...\n")

g = sigmoidGradient([1 -0.5 0 0.5 1])
@printf("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ")
@printf("%s", join(map(x -> @sprintf("%5f", x), g), " "))
@printf("\n\n")

@printf("Program paused. Press enter to continue.\n")
readline()

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

@printf("\nInitializing Neural Network Parameters ...\n")

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = [initial_Theta1[:] ; initial_Theta2[:]]


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
@printf("\nChecking Backpropagation... \n")

#  Check gradients by running checkNNGradients
checkNNGradients()

@printf("\nProgram paused. Press enter to continue.\n")
readline()


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

@printf("\nChecking Backpropagation (w/ Regularization) ... \n")

#  Check gradients by running checkNNGradients
lambda = 3
checkNNGradients(lambda)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size,
                             hidden_layer_size, num_labels, X, y, lambda)

@printf("""

  Cost at (fixed) debugging parameters (w/ lambda = 10): %f
  (this value should be about 0.576051)

""", debug_J)

@printf("Program paused. Press enter to continue.\n")

## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
@printf("\nTraining Neural Network... \n")

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.


# options = optimset('MaxIter', 50)
options = Opt(:LD_TNEWTON, length(initial_nn_params))
maxeval!(options, 50)

#  You should also try different values of lambda
lambda = 1

# Create "short hand" for the cost function to be minimized
function costFunction(theta, grad)
  cost, gs = nnCostFunction(theta, input_layer_size, hidden_layer_size,
                            num_labels, X, y, lambda)
  if length(grad) > 0
    grad[:] = gs
  end
  return cost
end

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
# [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

min_objective!(options, costFunction)
(cost, theta, _) = optimize(options, initial_nn_params)

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params[1:hidden_layer_size * (input_layer_size + 1)],
                 hidden_layer_size, (input_layer_size + 1))

Theta2 = reshape(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):end],
                 num_labels, (hidden_layer_size + 1))

@printf("Program paused. Press enter to continue.\n")
readline()

## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

@printf("\nVisualizing Neural Network... \n")

displayData(Theta1[:, 2:end])

@printf("\nProgram paused. Press enter to continue.\n")
readline()

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

@printf("\nTraining Set Accuracy: %f\n", mean(pred .== y) * 100)
