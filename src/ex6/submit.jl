export submit

include("../data.jl")
include("../submit.jl")

push!(LOAD_PATH, ".")

using PyCall, SVM, Spam

@pyimport scipy.io as si

function submit()
  parts = [
    Part(1, "Gaussian Kernel"),
    Part(2, "Parameters (C, sigma) for Dataset 3"),
    Part(3, "Email Preprocessing"),
    Part(4, "Email Feature Extraction")
  ]
  conf = Conf("support-vector-machines",
              "Support Vector Machines", parts, solver)

  submitWithConf(conf)
end

function solver(partId)
  # Random Test Cases
  x1 = sin(1:10)'
  x2 = cos(1:10)'
  ec = "the quick brown fox jumped over the lazy dog"
  wi = 1 + abs(round(x1 * 1863))
  wi = [wi ; wi]
  if partId == 1
    sim = gaussianKernel(x1, x2, 2)
    return @sprintf("%0.5f ", sim)
  elseif partId == 2
    data = si.loadmat("ex6data3.mat")
    X = data["X"]
    y = convert(Array{Int8, 2}, data["y"])
    Xval = data["Xval"]
    yval = convert(Array{Int8, 2}, data["yval"])

    C, sigma = dataset3Params(X, y, Xval, yval)
    out = @sprintf("%0.5f ", C)
    return out * @sprintf("%0.5f ", sigma)
  elseif partId == 3
    word_indices = processEmail(ec)
    println(join(map(x -> @sprintf("%d ", x), word_indices), " "))
    return join(map(x -> @sprintf("%d ", x), word_indices), " ")
  elseif partId == 4
    x = emailFeatures(wi)
    return join(map(y -> @sprintf("%d ", y), x), " ")
  end
end
