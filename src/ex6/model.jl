module SVMModel

export Model

type Model
  X :: Array{Float64}
  y :: Array{Any, 1}
  kernelFunction :: Function
  b :: Float64
  alphas :: Array{Float64, 1}
  w :: Array{Float64}
end

end
