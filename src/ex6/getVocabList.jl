@doc """
  GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
  cell array of the words
    vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
    and returns a cell array of the words in vocabList.
""" ->
function getVocabList()
  n = 1899
  # Total number of words in the dictionary

  #% Read the fixed vocabulary list

  # Store all dictionary words in cell array vocab{}

  # For ease of implementation, we use a struct to map the strings => integers
  # In practice, you'll want to use some form of hashmap
  vocabList = cell(n, 1)

  open("vocab.txt") do handler
    for line in eachline(handler)
      # Word Index (can ignore since it will be = i)
      i, word = split(line)
      # Actual Word
      vocabList[parse(Int, i)] = word
    end
  end

  return vocabList
end
