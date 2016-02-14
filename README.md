# Coursera Machine Learning in Julia

## Description
Scripts for [Coursera Stanford Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) assignments in Julia.  
As to exercises, this repository has only mock methods, so you should implement those first, and then submit the solutions.

## Requirements
Julia v0.4.x
scipy (to read .mat files)

You should also install some Julia libraries, as written in [REQUIRE](https://github.com/homuler/coursera-ml-julia/blob/master/REQUIRE).

## Usage
```shell
cd coursera-ml-julia/src/[exercise]
julia


julia> include("submit.jl") # when submitting
julia> submit()  

julia> include("ex1.jl") # when running exercise scripts

```

## ToDo
- Fix comments for Julia
- Migrate from PyPlot to Gadfly or Plotly
- Exercise7
  - displayData.jl
