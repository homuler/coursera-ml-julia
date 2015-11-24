export Part, Conf, UserInfo

type Part
  partId :: Int
  explanation :: AbstractString
end

type Conf
  assignmentSlug :: AbstractString
  itemName :: AbstractString
  parts :: Array{Part, 1}
  solver :: Function
end

type UserInfo
  email :: AbstractString
  token :: AbstractString
end
