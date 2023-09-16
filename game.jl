# A Bridge bidding attempt
# Kalle Prorok Stockholm Sweden sept 2023

#import Pkg; Pkg.add("CSV"); Pkg.add("DataFrames")
using CSV, DataFrames

using CommonRLInterface
using StaticArrays
using Crayons

const RL = CommonRLInterface

# To avoid episodes of unbounded length, we put an arbitrary limit to the length of an
# episode. Because time is not captured in the state, this introduces a slight bias in
# the value function.
const TESTEXAMPLES = 2000
const HAND_FEATURES = 5 # shape + hcp
const MAXBIDS = 6 # max no of bids in a row
const MAXBIDLEVEL = 16 # each bid can be 0 to MAXBIDLEVEL like 1 NT - pass(+0) or 1 NT - 3NT (+10)
#const DEALS = 200000 # no of deals to learn from, allocated in memory
const EPISODE_LENGTH_BOUND = MAXBIDS

startbid="1N" # Situation is a continuation after 1 Spade - 1 NT
suits=["♣", "♦", "♥", "♠", "N"]
rank=Dict([('♣',0), ('♦',1), ('♥',2), ('♠',3), ('N',4)] )
suitc=Dict([('♣','c'), ('♦','d'), ('♥','h'), ('♠','s'), ('N','n')] )
suc=['c','d','h','s','n']

global currenttestexample = 0

function n2b(num::Int)
  """
Convert a number to a bid-string. 0 gives Pass.  
  :param num: 
  :return: 
  """
  if num > 35
      num = 35  
  elseif num == 0
      return "P"
  end

  level = 1+(num-1) ÷ 5
  suit = suits[1+rem((num-1),5)]
  return string(level) * suit # * = Concatenate in Julia

end

function b2n(bid::String)
  """
Convert a str bid into a ranking number. P=0, 1C=1, 2C = 6..
  :param bid: 
  :return: 
  """
  if bid == "P"
      return 0

  elseif parse(Int32,bid[1]) > 7
      bid="7N"
  end
  ibid = parse(Int32,bid[1])
  return 1+(ibid-1)*5+rank[bid[2]]
end

const startbidn = b2n(startbid)

const TRICK_VAL=Dict([('♣',20), ('♦',20), ('♥',30), ('♠',30), ('N',30)] )

scoren=Matrix{Int16}(undef,36,14)

# Inspired by https://github.com/lorserker/ben/blob/main/src/scoring.py
function score(contract::String, is_vulnerable::Bool, n_tricks::Int)
  if contract == "P" return 0
  end
  level = codepoint(contract[1]) - codepoint('0')
  strain = contract[2]
  doubled = 'X' in contract
  redoubled = occursin("XX", contract)

  target = 6 + level

  final_score = 0
  if n_tricks >= target
      # contract made
      base_score = level * TRICK_VAL[strain]
      if strain == 'N'
          base_score += 10
      end    
      bonus = 0
      
      # doubles and redoubles
      if redoubled
          base_score *= 4
          bonus += 100
      elseif doubled
          base_score *= 2
          bonus += 50
      end
      # game bonus
      if base_score < 100
          bonus += 50
      else
          bonus += (is_vulnerable ? 500 : 300) # 500 if is_vulnerable else 300 end
      end
      # slam bonus
      if level == 6
          bonus += (is_vulnerable ? 750 : 500) #if is_vulnerable else 500 end
      elseif level == 7
          bonus += (is_vulnerable ? 1500 : 1000)# if is_vulnerable else 1000 end
      end
      n_overtricks = n_tricks - target
      overtrick_score = 0
      if redoubled
          overtrick_score = n_overtricks * (is_vulnerable ? 400 : 200)#(400 if is_vulnerable else 200)
      elseif doubled
          overtrick_score = n_overtricks * (is_vulnerable ? 200 : 100)# if is_vulnerable else 100)
      else
          overtrick_score = n_overtricks * TRICK_VAL[strain]
      end
      final_score = base_score + overtrick_score + bonus
  else
      # contract failed
      n_undertricks = target - n_tricks
      undertrick_values = []
      if is_vulnerable
         # undertrick_values = [100] * 13
          if redoubled
              undertrick_values = [400] + [600] * 12
          elseif doubled
              undertrick_values = [200] + [300] * 12
          end    
      else
#            undertrick_values = [50] * 13
          if redoubled
              undertrick_values = [200, 400, 400] + [600] * 10
          elseif doubled
              undertrick_values = [100, 200, 200] + [300] * 10
          end
      #final_score = -sum(undertrick_values[1:n_undertricks]) TODO - fix in Julia when opps can X/XX
      if is_vulnerable
          final_score = -100 * n_undertricks
      else
          final_score = -50 * n_undertricks
      end
      end 
  end   
  return final_score
end

function fill_scoren() # Make acess quicker via precalculations. 
  for contr=0:35
      for trick=0:13
          global scoren[contr+1,trick+1]=score(n2b(contr),false,trick)
      end
  end
end

function scoret(num::Int,tricks::Int)
  if num > 35 # more than 7NT
      return -10000
  else
       return scoren[num+1,tricks+1] # Handle pass and zero taken tricks = bad result ;)
  end
end     

function imp(diff::Int;topscore=false)
  if topscore
      return diff
  end        
  if diff < 0
      sgn=-1
      diff = -diff
  else
      sgn = 1
  end
  if diff<20
      return 0
  elseif diff < 50
      return sgn
  elseif diff < 90
      return sgn * 2
  elseif diff < 130
      return sgn * 3
  elseif diff < 170
      return sgn * 4
  elseif diff < 220
      return sgn * 5
  elseif diff < 270
      return sgn * 6
  elseif diff < 320
      return sgn * 7
  elseif diff < 370
      return sgn * 8
  elseif diff < 430
      return sgn * 9
  elseif diff < 500
      return sgn * 10
  elseif diff < 600
      return sgn * 11
  elseif diff < 750
      return sgn * 12
  elseif diff < 900
      return sgn * 13
  elseif diff < 1100
      return sgn * 14
  elseif diff < 1300
      return sgn * 15
  elseif diff < 1500
      return sgn * 16
  elseif diff < 1750
      return sgn * 17
  elseif diff < 2000
      return sgn * 18
  elseif diff < 2250
      return sgn * 19
  elseif diff < 2500
      return sgn * 20
  elseif diff < 3000
      return sgn * 21
  elseif diff < 3500
      return sgn * 22
  elseif diff < 4000
      return sgn * 23
  else
      return sgn * 24
  end
end

#impofs = 20 # offset to be added to imp-results to get positive scdfores during learning

print("Reading from file once..")
globdf = CSV.File("res220k.txt")
print(size(globdf))
 
mutable struct World <: AbstractEnv
  df::CSV.File
  ex::Int32
  bids::Int8
  state::MVector{HAND_FEATURES+MAXBIDS,Int8} # small size to save memory
end


function load_data(;path="res22k.txt")
  df = globdf # CSV.File(path)
  return df
end


function World()
  print("Loading data..")
  df = load_data()
  rows = size(df,1)

  ex = rand(2:(rows - TESTEXAMPLES))
  print("Calculating scores..selected ex:",ex)
  fill_scoren()
  bids = 0
  #rewards = Dict(
  #  SA[9,3] =>  10.0,
  #  SA[8,8] =>   3.0,
  #  SA[4,3] => -10.0,
  #  SA[4,6] =>  -5.0)

  return World(df,ex,bids,[df[ex][:sn],df[ex][:hn],df[ex][:dn],df[ex][:cn],df[ex][:pn],-1,-1,-1,-1,-1,-1])
end

#RL.reset!(env::World) = (env.state = SA[rand(1:env.size[1]), rand(1:env.size[2])])
#RL.actions(env::World) = [1:SA[1,0], SA[-1,0], SA[0,1], SA[0,-1]]
#RL.observe(env::World) = env.state

function RL.reset!(env::AlphaZero.Examples.BridgeBid.World) 
  rows = size(env.df,1)
  env.ex = rand(2:(rows - TESTEXAMPLES))
  if currenttestexample == 0
     global currenttestexample = rows - TESTEXAMPLES
  end
  env.ex = currenttestexample
  if currenttestexample < rows
     global currenttestexample = currenttestexample + 1
  end
  

#  env.ex = rand((rows - TESTEXAMPLES):rows) # For Validating
  ex = env.ex
#  print("Reset to ",ex," ")
  env.bids = 0
  env.state = MVector{HAND_FEATURES+MAXBIDS,Int8}(env.df[ex][:sn],env.df[ex][:hn],env.df[ex][:dn],env.df[ex][:cn],env.df[ex][:pn],-1,-1,-1,-1,-1,-1)
#  print("Reset ex:",ex," is:",env.state,"!")
end
  #state = SA[rand(1:env.size[1]), rand(1:env.size[2])])
#RL.actions(env::World) = [1:SA[1,0], SA[-1,0], SA[0,1], SA[0,-1]]
#RL.observe(env::AlphaZero.Examples.BridgeBid.World) = env.state


@provide RL.terminated(env::AlphaZero.Examples.BridgeBid.World) =  (env.bids > (MAXBIDS-1)) || (env.state[HAND_FEATURES+env.bids] == 0)  # TEmp

@provide RL.actions(env::AlphaZero.Examples.BridgeBid.World) =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

@provide RL.observe(env::AlphaZero.Examples.BridgeBid.World) = deepcopy(env.state)

function RL.act!(env::AlphaZero.Examples.BridgeBid.World, a::Integer)
  #ex = env.ex
  if (a < 0 || a > MAXBIDLEVEL)
    print("Error in action:",a)
    a = 0 # To continue run
  end
  if (a == 0) || (env.bids >=  (MAXBIDS-1)) # Temporary
    if env.bids <= (MAXBIDS-1)
      env.bids += 1
    end
    env.state[HAND_FEATURES+env.bids] = a
    bid = startbidn
    if env.bids > 0
       for i in range(HAND_FEATURES+1,HAND_FEATURES+env.bids)
          bid = bid+env.state[i] # Find out final bid
       end
    end

    suit = 1+rem((bid-1),5) # Find out in which suit or NT to see how many tricks can be taken
    if suit==1
      sc = :tcn
    elseif suit==2
      sc = :tdn 
    elseif suit==3
      sc = :thn
    elseif suit==4
      sc = :tsn
    elseif suit==5
      sc = :tnn
    else 
      print("ERROR:Strange suit in bid ",bid,suit)
      sc = :tnn # To continue run
    end
#    print("Sc:",sc,"Env.ex:",env.ex,"contr:",bid,"! ")
    return (imp(scoret(bid, env.df[env.ex][sc]) - env.df[env.ex][:parNS]))/10.0    # Calculate relative score as a reward -1..1
  else
    env.bids += 1
    env.state[HAND_FEATURES+env.bids] = a
#    print("bidno",env.bids,":",a," ")
    if (env.bids % 2) == 0 # North's time to bid, alternate hand but same player in AlphaZero
      env.state[1] = env.df[env.ex][:sn] # State is the shape and strength, more feautures can be added later
      env.state[2] = env.df[env.ex][:hn]
      env.state[3] = env.df[env.ex][:dn]
      env.state[4] = env.df[env.ex][:cn]
      env.state[5] = env.df[env.ex][:pn]
    else # South's time to bid
      env.state[1] = env.df[env.ex][:ss]
      env.state[2] = env.df[env.ex][:hs]
      env.state[3] = env.df[env.ex][:ds]
      env.state[4] = env.df[env.ex][:cs]
      env.state[5] = env.df[env.ex][:ps]
    end
  end

  return 0  #reward when no final bid is reached yet . get(env.rewards, env.state, 0.0)
end


@provide RL.player(env::AlphaZero.Examples.BridgeBid.World) = 1 # An MDP is a one player game
@provide RL.players(env::AlphaZero.Examples.BridgeBid.World) = [1]
@provide RL.observations(env::AlphaZero.Examples.BridgeBid.World) = env.state #SA[x, y] for x in 1:env.size[1], y in 1:env.size[2]]
@provide RL.clone(env::AlphaZero.Examples.BridgeBid.World) = World(env.df,env.ex, env.bids, deepcopy(env.state))
@provide RL.state(env::AlphaZero.Examples.BridgeBid.World) = deepcopy(env.state)
@provide RL.setstate!(env::AlphaZero.Examples.BridgeBid.World, s) = (env.state = deepcopy(s))
@provide RL.valid_action_mask(env::AlphaZero.Examples.BridgeBid.World) = (env.state[6] == -1 ? BitVector([1, 1, 1, 1,1,1, 1,1,1, 1,1,1,1, 1,1,1,1]) : BitVector([1, 1, 1, 1,1,1, 1,1,1, 1,1,1,1, 1,1,1,1]))
# Now allow pass as first bid, otherwise 0 at first item in  Bitvector

# Additional functions needed by AlphaZero.jl that are not present in 
# CommonRlInterface.jl. Here, we provide them by overriding some functions from
# GameInterface. An alternative would be to pass them as keyword arguments to
# CommonRLInterfaceWrapper.

#@provide RL.current_state(env::AlphaZero.Examples.BridgeBid.World) = deepcopy(env.state)

function GI.vectorize_state(env::AlphaZero.Examples.BridgeBid.World, state)
   v = zeros(Float32,HAND_FEATURES+MAXBIDS)
   for i in 1:(HAND_FEATURES+MAXBIDS)
      v[i] = env.state[i]/20.0
   end
   return v
end

function GI.render(env::AlphaZero.Examples.BridgeBid.World)
   io = open("result.csv","a")
   print("Hands:")
   if env.bids >= 1
      for i in 1:(HAND_FEATURES+env.bids)
         print(env.state[i], " ")
#         print(io,env.state[i], ",")
      end
      bid = startbidn
      bidp = startbidn
      for i in 1:env.bids
         bid += env.state[HAND_FEATURES+i]
         if env.state[HAND_FEATURES+i] == 0
           
           for j=1:(i-1)
              bidp += env.state[HAND_FEATURES+j]
              print(io,n2b(bidp),",")
           end 
           print(n2b(0)) #  Pass
           print(io,"Pass,")
           for j in i:(6-1)
               print(io,"no,") # pad with -1
           end 
           suit = 1+rem((bid-1),5) # Find out in which suit or NT to see how many tricks can be taken
           if suit==1
             sc = :tcn
           elseif suit==2
             sc = :tdn 
           elseif suit==3
             sc = :thn
           elseif suit==4
             sc = :tsn
           elseif suit==5
           sc = :tnn
          else 
            print("ERROR:Strange suit in bid ",bid,suit)
            sc = :tnn # To continue run
          end
#    print("Sc:",sc,"Env.ex:",env.ex,"contr:",bid,"! ")
          print(" ",scoret(bid, env.df[env.ex][sc]),'(', imp(scoret(bid, env.df[env.ex][sc]) - env.df[env.ex][:parNS]),')')    # Calculate relative score as a reward -1..1
          print(io,scoret(bid, env.df[env.ex][sc]),',', imp(scoret(bid, env.df[env.ex][sc]) - env.df[env.ex][:parNS]))    # Calculate relative score as a reward -1..1
          print(io,"\n")


         else
#           print(io,n2b(bid),",")
           print(n2b(bid))
         end
      end
   

        
   else 
 ##     io = open("result.csv","a")
      print(env.ex,":",env.df[env.ex][:north],"[",env.df[env.ex][:pn],"]",env.df[env.ex][:south],"[",env.df[env.ex][:ps],"]","(",env.df[env.ex][:parNS],")")
      print(io,env.ex,",",env.df[env.ex][:north],",",env.df[env.ex][:pn],",",env.df[env.ex][:south],",",env.df[env.ex][:ps],",",env.df[env.ex][:parNS],",")
      print(" no bids state: ")
      for i in 1:(HAND_FEATURES+MAXBIDS)
         print(env.state[i], " ")
         if i<= HAND_FEATURES
            print(io,env.state[i],",")
         end
      end  
      print(io,env.df[env.ex][:ss],",",env.df[env.ex][:hs],",",env.df[env.ex][:ds],",",env.df[env.ex][:cs],",",env.df[env.ex][:ps],",") # Also print South

   end
   print("\n")
 #  print(io,"\n")
   close(io)
end

const action_names = ["Pass","+1", "+2", "+3", "+4","+5", "+6", "+7", "+8","+9", "+10", "+11", "+12","+13","+14","+15","+16","+17","+18"]

function GI.action_string(env::AlphaZero.Examples.BridgeBid.World, a)
  idx = findfirst(==(a), RL.actions(env))
  return isnothing(idx) ? "?" : action_names[idx]
end

function GI.parse_action(env::AlphaZero.Examples.BridgeBid.World, s)
  idx = findfirst(==(s), action_names)
  return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

function GI.read_state(env::AlphaZero.Examples.BridgeBid.World)
  try
    s = split(readline())
    @assert length(s) == 1
    x = parse(Int, s[1])
#    y = parse(Int, s[2])
#    @assert 1 <= x <= env.size[1]
#    @assert 1 <= y <= env.size[2]
    return [env.df[x][:sn],env.df[x][:hn],env.df[x][:dn],env.df[x][:cn],env.df[x][:pn],-1,-1,-1,-1,-1,-1]
  catch e
    return nothing
  end
end


""" commented away
function GI.render(env::World)
  for y in reverse(1:env.size[2])
    for x in 1:env.size[1]
      s = SA[x, y]
      r = get(env.rewards, s, 0.0)
      if env.state == s
        c = ("+",)
      elseif r > 0
        c = (crayon"green", "o")
      elseif r < 0
        c = (crayon"red", "o")
      else
        c = (crayon"dark_gray", ".")
      end
      print(c..., " ", crayon"reset")
    end
    println("")
  end
end

function GI.vectorize_state(env::World, state)
  v = zeros(Float32, env.size[1], env.size[2])
  v[state[1], state[2]] = 1
  return v
end

const action_names = ["r", "l", "u", "d"]

function GI.action_string(env::World, a)
  idx = findfirst(==(a), RL.actions(env))
  return isnothing(idx) ? "?" : action_names[idx]
end

function GI.parse_action(env::World, s)
  idx = findfirst(==(s), action_names)
  return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

function GI.read_state(env::World)
  try
    s = split(readline())
    @assert length(s) == 2
    x = parse(Int, s[1])
    y = parse(Int, s[2])
    @assert 1 <= x <= env.size[1]
    @assert 1 <= y <= env.size[2]
    return SA[x, y]
  catch e
    return nothing
  end
end
"""
GI.heuristic_value(::AlphaZero.Examples.BridgeBid.World) = 0.0

GameSpec() = CommonRLInterfaceWrapper.Spec(AlphaZero.Examples.BridgeBid.World())
