module Spam

include("processEmail.jl")
include("emailFeatures.jl")
include("getVocabList.jl")

export processEmail, emailFeatures, getVocabList

end
