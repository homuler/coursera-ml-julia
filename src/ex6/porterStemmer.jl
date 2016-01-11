@doc """
  Applies the Porter Stemming algorithm as presented in the following
  paper:
  Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
    no. 3, pp 130-137

  Original code modeled after the C version provided at:
  http://www.tartarus.org/~martin/PorterStemmer/c.txt

  The main part of the stemming algorithm starts here. b is an array of
  characters, holding the word to be stemmed. The letters are in b[k0],
  b[k0+1] ending at b[k]. In fact k0 = 1 in this demo program (since
  matlab begins indexing by 1 instead of 0). k is readjusted downwards as
  the stemming progresses. Zero termination is not in fact used in the
  algorithm.

  To call this function, use the string to be stemmed as the input
  argument.  This function returns the stemmed word as a string.
""" ->
function porterStemmer(inString)

  # Lower-case string
  inString = lowercase(inString)

  global j
  b = inString
  k = length(b)
  k0 = 1
  j = k



  # With this if statement, strings of length 1 or 2 don't go through the
  # stemming process. Remove this conditional to match the published
  # algorithm.
  stem = b
  if k > 2
    # Output displays per step are commented out.
    # disp(sprintf('Word to stem: %s', b));
    x = step1ab(b, k, k0)
    # disp(sprintf('Steps 1A and B yield: %s', x{1}));
    x = step1c(x{1}, x{2}, k0)
    # disp(sprintf('Step 1C yields: %s', x{1}));
    x = step2(x{1}, x{2}, k0)
    # disp(sprintf('Step 2 yields: %s', x{1}));
    x = step3(x{1}, x{2}, k0)
    # disp(sprintf('Step 3 yields: %s', x{1}));
    x = step4(x{1}, x{2}, k0)
    # disp(sprintf('Step 4 yields: %s', x{1}));
    x = step5(x{1}, x{2}, k0)
    # disp(sprintf('Step 5 yields: %s', x{1}));
    stem = x{1}
  end
  return stem
end

# cons(j) is TRUE <=> b[j] is a consonant.
function cons(i, b, k0)
  c = true
  if b[i] in ['a', 'e', 'i', 'o', 'u']
    c = false;
  elseif b[i] == 'y'
    if i == k0
      c = true
    else
      c = ~cons(i - 1, b, k0)
    end
  end
  return c
end

# mseq() measures the number of consonant sequences between k0 and j.  If
# c is a consonant sequence and v a vowel sequence, and <..> indicates
# arbitrary presence,

#      <c><v>       gives 0
#      <c>vc<v>     gives 1
#      <c>vcvc<v>   gives 2
#      <c>vcvcvc<v> gives 3
#      ....
function measure(b, k0)
  global j
  n = 0
  i = k0

  while true
    if i > j
      return
    end
    if ~cons(i, b, k0)
      break
    end
    i += 1
  end
  i += 1

  while true
    while true
      if i > j
        return
      end
      if cons(i, b, k0)
        break
      end
      i += 1
    end

    i += 1
    n += 1

    while true
      if i > j
        return
      end
      if ~cons(i, b, k0)
        break
      end
      i += 1
    end
    i += 1
  end
  return m
end


# vowelinstem() is TRUE <=> k0,...j contains a vowel
function vowelinstem(b, k0)
  global j
  for i = k0:j
    if ~cons(i, b, k0)
      return true
    end
  end
  return false
end

#doublec(i) is TRUE <=> i,(i-1) contain a double consonant.
function doublec(i, b, k0)
  if i < k0+1
    return false
  end

  if b[i] ~= b[i - 1]
    return false
  end

  return cons(i, b, k0)
end


# cvc(j) is TRUE <=> j-2,j-1,j has the form consonant - vowel - consonant
# and also if the second c is not w,x or y. this is used when trying to
# restore an e at the end of a short word. e.g.
#
#      cav(e), lov(e), hop(e), crim(e), but
#      snow, box, tray.

function cvc(i, b, k0)
  if ((i < (k0+2)) || ~cons(i, b, k0) || cons(i-1, b, k0) || ~cons(i-2, b, k0))
    return false
  elseif (b(i) == 'w' || b(i) == 'x' || b(i) == 'y')
    return false
  end
  return true
end

# ends(s) is TRUE <=> k0,...k ends with the string s.
function ends(str, b, k)
  global j
  if (str[length(str)] != b[k])
    return false
  end # tiny speed-up
  if (length(str) > k)
    return false
  end

  if b[k-length(str)+1:k] == str
    j = k - length(str);
    return true
  end
  return false
end

# setto(s) sets (j+1),...k to the characters in the string s, readjusting
# k accordingly.

function setto(s, b, k)
  global j
  for i = j+1:(j+length(s))
    b[i] = s[i-j]
  end
  if k > j+length(s)
    b = b[j + length(s)] * b[(k+1):end]
  end
  k = length(b)
  return {b, k}
end

# rs(s) is used further down.
# [Note: possible null/value for r if rs is called]
function rs(str, b, k, k0)
  r = {b, k}
  if measure(b, k0) > 0
    r = setto(str, b, k)
  end
  return r
end
# step1ab() gets rid of plurals and -ed or -ing. e.g.

#       caresses  ->  caress
#       ponies    ->  poni
#       ties      ->  ti
#       caress    ->  caress
#       cats      ->  cat

#       feed      ->  feed
#       agreed    ->  agree
#       disabled  ->  disable

#       matting   ->  mat
#       mating    ->  mate
#       meeting   ->  meet
#       milling   ->  mill
#       messing   ->  mess

#       meetings  ->  meet

function step1ab(b, k, k0)
  global j
  if b[k] == 's'
    if ends("sses", b, k)
      k -= 2
    elseif ends("ies", b, k)
      retVal = setto('i', b, k);
      b = retVal{1}
      k = retVal{2}
    elseif (b[k-1] != 's')
      k -= 1
    end
  end
  if ends("eed", b, k)
    if measure(b, k0) > 0
      k -= 1
    end
  elseif (ends("ed", b, k) || ends("ing", b, k)) && vowelinstem(b, k0)
    k = j
    retVal = {b, k}
    if ends("at", b, k)
      retVal = setto("ate", b[k0:k], k)
    elseif ends("bl", b, k)
      retVal = setto("ble", b[k0:k], k)
    elseif ends("iz", b, k)
      retVal = setto("ize", b[k0:k], k)
    elseif doublec(k, b, k0)
      retVal = {b, k-1}
      if b[retVal{2}] == 'l' || b(retVal{2}) == 's' || b(retVal{2}) == 'z'
        retVal = {retVal{1}, retVal{2}+1}
      end
    elseif measure(b, k0) == 1 && cvc(k, b, k0)
      retVal = setto("e", b(k0:k), k)
    end
    k = retVal{2}
    b = retVal{1}(k0:k)
  end
  j = k
  return {b[k0:k], k}
end

#  step1c() turns terminal y to i when there is another vowel in the stem.
function step1c(b, k, k0)
  global j
  if ends("y", b, k) && vowelinstem(b, k0)
    b[k] = 'i'
  end
  j = k
  return {b, k}
end

# step2() maps double suffices to single ones. so -ization ( = -ize plus
# -ation) maps to -ize etc. note that the string before the suffix must give
# m() > 0.
function step2(b, k, k0)
  global j
  s2 = {b, k}
  if b[k-1] == {'a'}
    if ends("ational", b, k)
      s2 = rs("ate", b, k, k0)
    elseif ends("tional", b, k)
      s2 = rs("tion", b, k, k0)
    end
  elseif b[k-1] == {'c'}
    if ends("enci", b, k)
      s2 = rs("ence", b, k, k0)
    elseif ends("anci", b, k)
      s2 = rs("ance", b, k, k0)
    end
  elseif b[k-1] == {'e'}
    if ends("izer", b, k)
      s2 = rs("ize", b, k, k0)
    end
  elseif b[k-1] == {'l'}
    if ends("bli", b, k)
      s2 = rs("ble", b, k, k0)
    elseif ends("alli", b, k)
      s2 = rs("al", b, k, k0)
    elseif ends("entli", b, k)
      s2 = rs("ent", b, k, k0)
    elseif ends("eli", b, k)
      s2 = rs("e", b, k, k0)
    elseif ends("ousli", b, k)
      s2 = rs("ous", b, k, k0)
    end
  elseif b[k-1] == case {'o'}
    if ends("ization", b, k)
      s2 = rs("ize", b, k, k0)
    elseif ends("ation", b, k)
      s2 = rs("ate", b, k, k0)
    elseif ends("ator", b, k)
      s2 = rs("ate", b, k, k0)
    end
  elseif b[k-1] == {'s'}
    if ends("alism", b, k)
      s2 = rs("al", b, k, k0)
    elseif ends("iveness", b, k)
      s2 = rs("ive", b, k, k0)
    elseif ends("fulness", b, k)
      s2 = rs("ful", b, k, k0)
    elseif ends("usness", b, k)
      s2 = rs("ous", b, k, k0)
    end
  elseif b[k-1] == {'t'}
    if ends("aliti", b, k)
      s2 = rs("al", b, k, k0)
    elseif ends("iviti", b, k)
      s2 = rs("ive", b, k, k0)
    elseif ends("biliti", b, k)
      s2 = rs("ble", b, k, k0)
    end
  elseif b[k-1] == {'g'}
    if ends("logi", b, k)
      s2 = rs("log", b, k, k0)
    end
  end
  j = s2{2}
  return s2
end
# step3() deals with -ic-, -full, -ness etc. similar strategy to step2.
function step3(b, k, k0)
  global j
  s3 = {b, k}
  if ends("icate", b, k)
  if b[k] == {'e'}
      s3 = rs("ic", b, k, k0)
    elseif ends("ative", b, k)
      s3 = rs("", b, k, k0)
    elseif ends("alize", b, k)
      s3 = rs("al", b, k, k0)
    end
  elseif b[k] == {'i'}
    if ends("iciti", b, k)
      s3 = rs("ic", b, k, k0)
    end
  elseif b[k] == {'l'}
    if ends("ical", b, k)
      s3 = rs("ic", b, k, k0)
    elseif ends("ful", b, k)
      s3 = rs("", b, k, k0)
    end
  elseif b[k] == {'s'}
    if ends("ness", b, k)
      s3 = rs("", b, k, k0)
    end
  end
  j = s3{2}
  return s3
end

# step4() takes off -ant, -ence etc., in context <c>vcvc<v>.
function step4(b, k, k0)
  global j
  if b[k-1] == {'a'}
    if ends("al", b, k)
    end
  elseif b[k-1] == {'c'}
    if ends("ance", b, k)
    elseif ends("ence", b, k)
    end
  elseif b[k-1] == {'e'}
    if ends("er", b, k)
    end
  elseif b[k-1] == {'i'}
    if ends("ic", b, k)
    end
  elseif b[k-1] == {'l'}
    if ends("able", b, k)
    elseif ends("ible", b, k)
    end
  elseif b[k-1] == {'n'}
    if ends("ant", b, k)
    elseif ends("ement", b, k)
    elseif ends("ment", b, k)
    elseif ends("ent", b, k)
    end
  elseif b[k-1] == {'o'}
    if ends("ion", b, k)
      if j == 0
      elseif ~(strcmp(b[j],'s') || strcmp(b[j],'t'))
        j = k
      end
    elseif ends("ou", b, k)
    end
  elseif b[k-1] == {'s'}
    if ends("ism", b, k)
    end
  elseif b[k-1] == {'t'}
    if ends("ate", b, k)
    elseif ends("iti", b, k)
    end
  elseif b[k-1] == {'u'}
    if ends("ous", b, k)
    end
  elseif b[k-1] == {'v'}
    if ends("ive", b, k)
    end
  elseif b[k-1] == {'z'}
    if ends("ize", b, k)
    end
  end
  if measure(b, k0) > 1
    s4 = {b[k0:j], j}
  else
    s4 = {b[k0:k], k}
  end
  return s4
end

# step5() removes a final -e if m() > 1, and changes -ll to -l if m() > 1.
function step5(b, k, k0)
  global j
  j = k
  if b[k] == 'e'
    a = measure(b, k0)
    if (a > 1) || ((a == 1) && ~cvc(k-1, b, k0))
      k -= 1
    end
  end
  if (b[k] == 'l') && doublec(k, b, k0) && (measure(b, k0) > 1)
    k -= 1
  end
  return {b[k0:k], k}
end
