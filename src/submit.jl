using Requests
using JSON

type Part
  partId :: Int
  sourceFiles :: Array{AbstractString, 1}
  explanation :: AbstractString
  solver :: Function
end

type Conf
  assignmentSlug :: AbstractString
  itemName :: AbstractString
  parts :: Array{Part, 1}
end

type UserInfo
  email :: AbstractString
  token :: AbstractString
end

type SubmissionReqBody
  assignmentSlug :: AbstractString
  submitterEmail :: AbstractString
  secret :: AbstractString
  parts :: Dict{Int, Dict{AbstractString, AbstractString}}
end

function submit(conf :: Conf)
  @printf "== Submitting solutions | %s...\n" conf.itemName
  userInfo = loadToken("token.mat")
  show(userInfo)
  if length(userInfo.token) == 0
    println("!! Submission Cancelled")
    return end
  postData = SubmissionReqBody(conf.assignmentSlug, userInfo.email, userInfo.token, solveProblems(conf.parts))
  resp = Requests.post("https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1", data = {"jsonBody" => JSON.json(postData)})
  return resp
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
  end
end

function promptToken()
  print("Login (email address): ");
  email = chomp(readline(STDIN))
  print("Token: ");
  token = chomp(readline(STDIN))
  UserInfo(email, token)
end

function makePostBody(conf :: Conf, userInfo :: UserInfo)
  body = SubmissionReqBody(conf.assignmentSlug,
                           userInfo.email,
                           userInfo.token,
                           solveProblems(conf))
  return JSON.json(body)
end

function solveProblems(parts :: Array{Part, 1})
  anslist = Dict{Int, Dict{AbstractString, AbstractString}}()
  for part in parts
    anslist[part.partId] = { "output" => part.solver() }
  end
  return anslist
end
# submit(Conf("foo", "bar", [Part(1, ["baz"], "tux", convert)]))
#conf = Conf("foo", "bar", [Part(1, ["baz"], "hoge", () -> "Hello")])
#submit(conf)
