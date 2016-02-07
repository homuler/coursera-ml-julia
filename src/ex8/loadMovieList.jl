@doc """
  GETMOVIELIST reads the fixed movie list in movie.txt and returns a
  cell array of the words
    movieList = GETMOVIELIST() reads the fixed movie list in movie.txt
    and returns a cell array of the words in movieList.
""" ->
function loadMovieList()

  # Store all movies in cell array movie{}
  n = 1682  # Total number of movies

  movieList = cell(n, 1)
  open("movie_ids.txt") do handler
    for line in eachline(handler)
      # Word Index (can ignore since it will be = i)
      movieInfo = split(line)
      # Actual Word
      movieList[parse(Int, movieInfo[1])] = join(movieInfo[2:end], " ")
    end
  end

  return movieList
end
