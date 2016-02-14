## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.jl
#     dataset3Params.jl
#     processEmail.jl
#     emailFeatures.jl
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
push!(LOAD_PATH, ".")

using PyCall, SVM, Spam

@pyimport scipy.io as si

## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

@printf("\nPreprocessing sample email (emailSample1.txt)\n")

# Extract Features

open("emailSample1.txt") do handler
  file_contents = readall(handler)
  global word_indices = processEmail(file_contents)

  @printf("Word Indices: \n")
  @printf("%s", join(map(x -> @sprintf("%d", x), word_indices), " "))
  @printf("\n\n")

  @printf("Program paused. Press enter to continue.\n")
end

# Print Stats

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

@printf("\nExtracting features from sample email (emailSample1.txt)\n")

# Extract Features
open("emailSample1.txt") do handler
  file_contents = readall(handler)
  word_indices = processEmail(file_contents)
  features      = emailFeatures(word_indices)

  @printf("Length of feature vector: %d\n", length(features))
  @printf("Number of non-zero entries: %s\n",
            join(map(x -> @sprintf("%d", x), sum(features .> 0)), " "))

  @printf("Program paused. Press enter to continue.\n")
end

# Print Stats

readline()

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = si.loadmat("spamTrain.mat")
X = data["X"]
y = convert(Array{Int8, 2}, data["y"])

@printf("\nTraining Linear SVM (Spam Classification)\n")
@printf("(this may take 1 to 2 minutes) ...\n")

C = 0.1
model = svmTrain(X, y, C, linearKernel)

p = svmPredict(model, X)

@printf("Training Accuracy: %f\n", mean(p .== y) * 100)

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
data = si.loadmat("spamTest.mat")
Xtest = data["Xtest"]
ytest = convert(Array{Int8, 2}, data["ytest"])

@printf("\nEvaluating the trained Linear SVM on a test set ...\n")

p = svmPredict(model, Xtest)

@printf("Test Accuracy: %f\n", mean(p .== ytest) * 100)


## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list
vs = sort(collect(enumerate(model.w)), lt=(a, b) -> a[2] > b[2])
weight = map(x -> x[2], vs)
idx = map(x -> x[1], vs)
vocabList = getVocabList()

@printf("\nTop predictors of spam: \n")
for i = 1:15
  @printf(" %-15s (%f) \n", vocabList[idx[i]], weight[i])
end

@printf("\n\n")
@printf("\nProgram paused. Press enter to continue.\n")
readline()

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = "spamSample1.txt"

# Read and predict
open(filename) do handler
  file_contents = readall(handler)
  word_indices = processEmail(file_contents)
  x = emailFeatures(word_indices)
  p = svmPredict(model, x)

  @printf("\nProcessed %s\n\nSpam Classification: %s\n", filename, join(map(x -> @sprintf("%d", x), p), " "))
  @printf("(1 indicates spam, 0 indicates not spam)\n\n")
end
