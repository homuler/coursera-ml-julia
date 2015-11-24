using Requests
using JSON

export submitWithConf

include("data.jl")

type SubmissionReqBody
  assignmentSlug :: AbstractString
  submitterEmail :: AbstractString
  secret :: AbstractString
  parts :: Dict{Int, Dict{AbstractString, AbstractString}}
end

function submitWithConf(conf :: Conf)
  @printf "== Submitting solutions | %s...\n" conf.itemName
  userInfo = loadToken("token.mat")
  if length(userInfo.token) == 0
    println("!! Submission Cancelled")
    return end
  postData = SubmissionReqBody(conf.assignmentSlug, userInfo.email, userInfo.token, solveProblems(conf))

  try
    resp = Requests.post("https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1",
                         data = Dict("jsonBody" => JSON.json(postData)))
    respJSON = Requests.json(resp)
    if haskey(respJSON, "errorMessage")
      @printf "!! Submission failed: %s\n" respJSON["errorMessage"]
    else
      showFeedback(conf.parts, respJSON)
      write(open("token.mat", "w"), @sprintf("%s\n%s\n", userInfo.email, userInfo.token))
    end
    return resp
  catch exp
    @printf "!! Submission failed: unexpected error\n"
    @printf "!! Please try again later.\n"
    rethrow(exp)
  end
end

function loadToken(path :: AbstractString)
  validateToken(arr) = length(arr) == 2

  if isreadable(path)
    open(path) do handler
      lines = map(chomp, readlines(handler))
      if validateToken(lines)
        UserInfo(lines[1], lines[2])
      else
        println("Invalid Token: Token file's format is incorrect. path -> " * path)
      end
    end
  else
    return promptToken()
  end
end

function promptToken()
  print("Login (email address): ");
  email = chomp(readline(STDIN))
  print("Token: ");
  token = chomp(readline(STDIN))
  UserInfo(email, token)
end

function solveProblems(conf :: Conf)
  anslist = Dict{Int, Dict{AbstractString, AbstractString}}()
  for part in conf.parts
    println(conf.solver(part.partId))
    anslist[part.partId] = Dict("output" => conf.solver(part.partId))
  end
  return anslist
end

function showFeedback(parts :: Array{Part, 1}, respJSON :: Dict)
  @printf "== \n"
  @printf "== %43s | %9s | %-s\n" "Part Name" "Score" "Feedback"
  @printf "== %43s | %9s | %-s\n" "---------" "-----" "--------"
  for part in parts
    partEval = respJSON["partEvaluations"][string(part.partId)]
    partFeedback = respJSON["partFeedbacks"][string(part.partId)]
    score = @sprintf("%d / %3d", partEval["score"], partEval["maxScore"])
    @printf "== %43s | %9s | %-s\n" part.explanation score partFeedback
  end
  eval = respJSON["evaluation"]
  totalScore = @sprintf("%d / %d", eval["score"], eval["maxScore"])
  @printf "==                                   --------------------------------\n"
  @printf "== %43s | %9s | %-s\n" "" totalScore ""
  @printf "== \n"
end
# submit(Conf("foo", "bar", [Part(1, ["baz"], "tux", convert)]))
#conf = Conf("foo", "bar", [Part(1, ["baz"], "hoge", () -> "Hello")])
#submit(conf)
