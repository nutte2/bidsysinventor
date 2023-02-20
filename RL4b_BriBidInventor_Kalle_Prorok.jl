# Reinforcemement Learning(RL) for Bridge Bidding
# Using Tabular Q-learning with shape and bidding history as state and bids (incl Pass) as actions
# V0.2 Kalle.Prorok@gmail.com Stockholm 2023-02-19
# Now max 4 bids.

import Pkg; 
Pkg.add("CSV")
using CSV

#Pkg.add("BenchmarkTools")
#using BenchmarkTools

using Random: seed!
using Serialization
using Dates
# --- Bridge part ---
# ♠♥♦♣

startbid="1N" # Situation is a continuation after 1 Spade - 1 NT
suits=["♣", "♦", "♥", "♠", "N"]
rank=Dict([('♣',0), ('♦',1), ('♥',2), ('♠',3), ('N',4)] )
suitc=Dict([('♣','c'), ('♦','d'), ('♥','h'), ('♠','s'), ('N','n')] )
suc=['c','d','h','s','n']

const n_bids = 35 # Last bid = 7NT
const TESTEXAMPLES = 2000 # Save some of the last examples for evaluation
const TESTEPISODES = 1000 # Number of last episodes to use for calculating average reward during training
const PLOTLAST = 24 # Show last examples during training
const n_actions = 11 #16
const HANDFEATURES = 4 # The number of features. i.e. Spades,Hearts,Diamonds, H.c.p. (Clubs implicit). LTC, controls etc can be added
const q0 = 0 # Init value in Q-tables. High value encourages exploration (search) for promising alternatives
const EPISODES = 150_000_000 # 30 miljons take ca 30 cpu-mins (using 6 cores for different Learning rates).

const EPS_START = 0.5 # Control the exploration, a lot in the beginning and less later
const EPS_END = 0.02 # Final eps value; probability of a random bid at end of learning
const EPS_DECAY = (EPISODES/10) # the episode when EPS_START is multiplied by exp(-1)

scoren=Matrix{Int16}(undef,36,14)
#print(sizeof(scoren))
#print(typeof(scoren))

rewardlog=zeros(10,EPISODES÷1_000_000) # Max 8 processes/threads for storing avg rewards

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

function imp(diff::Int;topscore=true)
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

function load_data(;path="res22k.txt")
    df = CSV.File(path)

    return df
end

function reduce!(df::CSV.File) # Reduce number of states via hcp only even numbers (round downwards). Didn't work..
    rows = size(df,1)
    println(df[3][9]," ",df[4][9], " ",df[3][16]," ",df[4][16])
    for ex in range(2,rows)
        df[ex][9] = (df[ex][9] ÷ 2) * 2 # Round 11 down to 10
        df[ex][16] = (df[ex][16] ÷ 2) * 2 # Round 11 down to 10
    end
    println(df[3][9]," ",df[4][9], " ",df[3][16]," ",df[4][16])
end

# ----- RL/Q-learning part -----

function env_reset!(df::CSV.File,Q::Dict{Vector{Int8}, Vector{Float16}})
    rows = size(df,1)
    ex = rand(2:(rows - TESTEXAMPLES)) #Select one of the examples. Save some for test afterwards. Index 1 is maybe titles/labels
    state = convert(Vector{Int8},[df[ex][:sn],df[ex][:hn],df[ex][:dn],(df[ex][:pn] ÷ 2) * 2]) # Clubs indirectly via 13-. Points grouped in 8,10,12,14,..
    if !haskey(Q,state) # New state, initialize action-values (with zero or maybe with q0 to inspire for exploration?)
        Q[state]=ones(Int8, n_actions)*q0
    end
    return state, ex
end

function env_step!(df::CSV.File, ex::Int, Q::Dict{Vector{Int8}, Vector{Float16}}, state::Vector{Int8}, action::Int)
    if length(state) == HANDFEATURES # North has selected 1st bid, new state will be south's hand with N's selected bid (action)
        newstate = convert(Vector{Int8},[df[ex][:ss],df[ex][:hs],df[ex][:ds],(df[ex][:ps] ÷ 2) * 2, action-1]) # Clubs indirectly via 13-. Points grouped in 8,10,12,14,..
        if !haskey(Q,newstate) # New state, initialize action-values (with zero or maybe with q0 to inspire for exploration?)
            Q[newstate]=ones(Int8, n_actions)*q0
        end
        reward = 0
        done = false
    elseif length(state) == HANDFEATURES + 1 # South have bid
        nbid = state[HANDFEATURES + 1] # Take care of what N did bid last time(N first bid)
        if nbid == 0 # N passes
#            contract = n2b(startbidn) 

            newstate = Nothing # No state follows pass
            #print("c2=",contract[2],"r=",rank)
#            sym = Symbol("t",suitc[contract[2]],"n") # Find where to look for no of tricks via DDS, for now: assume N is declarer
            sym = Symbol("t",suc[1+rem(startbidn-1,5)],"n") # Find where to look for no of tricks via DDS, for now: assume N is declarer
            tricks = df[ex][sym] # TODO: find correct declarer, now assume North
#            reward = imp(score(contract,false, tricks) - df[ex][:parNS]) # Use relative score compared to par(best) result and exaluate in imp (or topscore)
            reward = imp(scoret(startbidn, tricks) - df[ex][:parNS]) # Use relative score compared to par(best) result and exaluate in imp (or topscore)
    #        print("L state=",length(state)," C:",contract, " T:", tricks," in sym:",sym," bids:",b2n(startbid)," 5: ",state[5]," 6: ",state[6]," ")
    #        env_render(df::CSV.File,ex::Int,reward::Int64)
            done = true
    
        else
            newstate = convert(Vector{Int8},[df[ex][:sn],df[ex][:hn],df[ex][:dn],(df[ex][:pn] ÷ 2) * 2,nbid,action-1]) # North hand again. Action=1 if S passes stored as 0
            if !haskey(Q,newstate) # New state, initialize action-values (with zero or maybe with q0 to inspire for exploration?)
                Q[newstate]=ones(Int8, n_actions)*q0
            end  
            reward = 0
            done = false
        end    
    elseif length(state) == HANDFEATURES + 2 # South have bid
        nbid = state[HANDFEATURES + 1] # Take care of what N did bid last time(N first bid)
        sbid = state[HANDFEATURES + 2] # Take care of what S did bid last time(N first bid)
        if sbid == 0 # S passes
            #contract = n2b(startbidn+nbid) 
            newstate = Nothing # No state follows pass
            #print("c2=",contract[2],"r=",rank)
            sym = Symbol("t",suc[1+rem(startbidn-1+nbid,5)],"n")#suitc[contract[2]],"n") # Find where to look for no of tricks via DDS, for now: assume N is declarer
            #print(sym)
            tricks = df[ex][sym] # TODO: find correct declarer, now assume North
#            reward = imp(score(contract,false, tricks) - df[ex][:parNS]) # Use relative score compared to par(best) result and exaluate in imp (or topscore)
            reward = imp(scoret(startbidn+nbid, tricks) - df[ex][:parNS]) # Use relative score compared to par(best) result and exaluate in imp (or topscore)
    #        print("L state=",length(state)," C:",contract, " T:", tricks," in sym:",sym," bids:",b2n(startbid)," 5: ",state[5]," 6: ",state[6]," ")
    #        env_render(df::CSV.File,ex::Int,reward::Int64)
            done = true
    
        else
            newstate = convert(Vector{Int8},[df[ex][:ss],df[ex][:hs],df[ex][:ds],(df[ex][:ps] ÷ 2) * 2, nbid, sbid, action-1])
            if !haskey(Q,newstate) # New state, initialize action-values (with zero or maybe with q0 to inspire for exploration?)
                Q[newstate]=ones(Int8, n_actions)*q0
            end  
            reward = 0
            done = false
        end    
    elseif length(state) == HANDFEATURES + 3 # South have bid
        nbid = state[HANDFEATURES + 1] # Take care of what N did bid last time(N first bid)
        sbid = state[HANDFEATURES + 2] # Take care of what S did bid last time(N first bid)
        nrebid = state[HANDFEATURES + 3]
        if nrebid == 0 # N passes
#            contract = n2b(startbidn+nbid+sbid) 
            newstate = Nothing # No state follows pass
            #print("c2=",contract[2],"r=",rank)
#            sym = Symbol("t",suitc[contract[2]],"n") # Find where to look for no of tricks via DDS, for now: assume N is declarer
            sym = Symbol("t",suc[1+rem(startbidn-1+nbid+sbid,5)],"n") # Find where to look for no of tricks via DDS, for now: assume N is declarer
            #print(sym)
            tricks = df[ex][sym] # TODO: find correct declarer, now assume North
#            reward = imp(score(contract,false, tricks) - df[ex][:parNS]) # Use relative score compared to par(best) result and exaluate in imp (or topscore)
            reward = imp(scoret(startbidn+nbid+sbid, tricks) - df[ex][:parNS]) # Use relative score compared to par(best) result and exaluate in imp (or topscore)
    #        print("L state=",length(state)," C:",contract, " T:", tricks," in sym:",sym," bids:",b2n(startbid)," 5: ",state[5]," 6: ",state[6]," ")
    #        env_render(df::CSV.File,ex::Int,reward::Int64)
            done = true
    
        else
            newstate = convert(Vector{Int8},[df[ex][:sn],df[ex][:hn],df[ex][:dn],(df[ex][:pn] ÷ 2) * 2, nbid, sbid, nrebid, action-1])
            if !haskey(Q,newstate) # New state, initialize action-values (with zero or maybe with q0 to inspire for exploration?)
                Q[newstate]=ones(Int8, n_actions)*q0
            end  
            reward = 0
            done = false
        end    
        
    else # Both North and South has bidden twice, now terminate via autopass. More bid-rounds can be added between. No matter what action was selected now.
        #contract = n2b(startbidn + state[HANDFEATURES + 1] + state[HANDFEATURES + 2] + state[HANDFEATURES + 3] + state[HANDFEATURES + 3])#action - 1) # Add N's bid + S's bid (S action 0 = pass)
        contrn = startbidn + state[HANDFEATURES + 1] + state[HANDFEATURES + 2] + state[HANDFEATURES + 3] + state[HANDFEATURES + 4] 
        newstate = Nothing # No state follows autopass
        #print("c2=",contract[2],"r=",rank)
#        sym = Symbol("t",suitc[contract[2]],"n") # Find right DDS-trick-suit. Fix declarer in next version
        sym = Symbol("t",suc[1+rem(contrn-1,5)],"n") # Find right DDS-trick-suit. Fix declarer in next version
        #print(sym)
        tricks = df[ex][sym] # TODO: find correct declarer, now assume North
        reward = imp(scoret(contrn, tricks) - df[ex][:parNS]) # Use relative score compared to par(best) result and exaluate in imp (or topscore)
#        print("L state=",length(state)," C:",contract, " T:", tricks," in sym:",sym," bids:",b2n(startbid)," 5: ",state[5]," 6: ",state[6]," ")
#        env_render(df::CSV.File,ex::Int,reward::Int64)
        done = true
    end
    return newstate, reward, done, ex
end

function env_render(df::CSV.File,ex::Int,reward::Int64) # Show hands
    print(df[ex][:north]," ",df[ex][:sn],df[ex][:hn],df[ex][:dn],df[ex][:cn],'(',df[ex][:pn],") ")
    println(df[ex][:south]," ",df[ex][:ss],df[ex][:hs],df[ex][:ds],df[ex][:cs],'(',df[ex][:ps],") ",reward," [", df[ex][:parNS], "]")
end

function action_dist_sample(epsilon::Float64, ainstate::Vector{Float16})
    # TODO; restrict action to keep final contract at or below 7NT (35)
    if length(ainstate) > 0
        am = argmax(ainstate) # Find no of best action
        li = [am]
        for i=1:n_actions
            if abs(ainstate[i] - ainstate[am]) < 0.001 # Maybe more actions have the same score?
                if i != am
                    push!(li,i)  
                end
            end
        end
        action_star = rand(li) # Select best action randomly among them

        if rand() < epsilon # epsilon-greedy selection
            action = rand(1:n_actions)
        else
            action = action_star
        end
    else
        action = 1
        action_star = 1
        print("ads:No vec?")
    end    
    return action, action_star
end

function max_all_a(Q::Dict{Vector{Int8}, Vector{Float16}}, newstate::Vector{Int8}) 
    best_Q = maximum(Q[newstate]) # Find highest values for all possible action in the new state
    return best_Q
end

function state2bidseq(state::Vector{Int8}) # Show state as a string with n_bids
    if length(state) == HANDFEATURES+1
        str = n2b(startbidn + state[HANDFEATURES+1])
    elseif length(state) == HANDFEATURES+2
        if state[HANDFEATURES+2] > 0
            str = n2b(startbidn + state[HANDFEATURES+1]) * n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2])
        else
            str = n2b(startbidn + state[HANDFEATURES+1]) * "  "
        end
    elseif length(state) == HANDFEATURES+3
        if state[HANDFEATURES+3] > 0
            str = n2b(startbidn + state[HANDFEATURES+1]) *  n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2]) * n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2] + state[HANDFEATURES+3])
        else
            str = n2b(startbidn + state[HANDFEATURES+1]) * n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2])
        end
    elseif length(state) == HANDFEATURES+4
        if state[HANDFEATURES+4] > 0
           str = n2b(startbidn + state[HANDFEATURES+1]) *  n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2]) 
           str = str * n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2] + state[HANDFEATURES+3]) * n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2] + state[HANDFEATURES+3]+ state[HANDFEATURES+4])
        else
            str = n2b(startbidn + state[HANDFEATURES+1]) *  n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2]) * n2b(startbidn + state[HANDFEATURES+1] + state[HANDFEATURES+2] + state[HANDFEATURES+3])
        end
    else
        str = "s2b:strange"
    end    
    return str
end

function run_episode!(Q::Dict{Vector{Int8}, Vector{Float16}}, df::CSV.File, episode::Int, lr::Float64, epsilon::Float64, gamma::Float64, n_actions::Int, lambda::Float64)

    state,ex = env_reset!(df,Q)
    reward = 0
    terminated = false
    action, action_star = action_dist_sample(epsilon, Q[state])
    salist=[(state,action, action_star)] # Eligibility traces - keep track of full auction sequence from start to update backwards
    while !terminated
        state_prim, reward, done, ex = env_step!(df, ex, Q, state, action)
        if done
            delta = reward - Q[state][action] # Q[state_prim] does not exist in final state
            if episode > (EPISODES-PLOTLAST) # (First) and last 
                print(state2bidseq(state)," ")
                env_render(df,ex,reward)
            end 
            action_prim = action
            action_star = action # might be troublesome if last action was exploratory?
            terminated = true
        else
            action_prim, action_star = action_dist_sample(epsilon, Q[state_prim])
            delta = reward + gamma * Q[state_prim][action_prim] - Q[state][action_star]
        end
        #e[state][action] += 1
        #forall states,action
        e = 1
        for i = length(salist):-1:1 # Update previously visited states (eligibility traces) backwards
           st,ac,as = salist[i] 
           Q[st][ac] += lr * delta * e # last states gets highets updates, e decreases exponentially
           if ac == as # Not an exploratory (random) action; just update normal exploiting actions
           e *= gamma * lambda
           else
                e = 0 # Don't update from exploratory moves (this could maybe be bad)     
           end
        end
        if !terminated
            push!(salist,(state_prim, action_prim, action_star)) # Let those be with in the next update
        end
        state = state_prim
        action = action_prim   
    end

"""
#--- Without eligibility traces:

    state,ex = env_reset!(df,Q)
    reward = 0
    terminated = false
    while !terminated
        action = action_dist_sample(epsilon, Q[state])
        newstate, reward, done, ex = env_step!(df, ex, Q, state, action)
        if done
            Q[state][action] += lr * (reward - Q[state][action] ) # Q-learning
            if episode > (EPISODES-PLOTLAST) # (First) and last 
                print(state2bidseq(state)," ")
                env_render(df,ex,reward)
            end    
            # state,ex = env_reset(df)
            terminated = true
        else
            #max = max_all_a(Q,newstate)
#            print("max=",max,typeof(max)," Qsa=", Q[state][action],typeof(Q[state][action]),typeof(action), "r=",reward,typeof(reward),"g=",gamma,typeof(gamma),"lr=",lr,typeof(lr))
#           Q[state][action] = Q[state][action] + lr * (reward + gamma * max - Q[state][action] )
            Q[state][action] +=  lr * (reward + gamma * maximum(Q[newstate]) - Q[state][action] )  # Reward is usually 0..
            state = newstate
        end
    end
"""    
    return reward
end

function train!(Q::Dict{Vector{Int8}, Vector{Float16}} ,df::CSV.File, episodes::Int, lr::Float64, epsilon::Float64, gamma::Float64, n_actions::Int, i::Int, lambda::Float64)
    rewards=0
    f = exp(-1 / EPS_DECAY) # Faster calculation once than exp every loop
    et = 1
    rew = 0
    for episode in range(1,episodes)
        et *= f
#        eps_threshold = EPS_END + (epsilon - EPS_END) * exp(-1. * episode / EPS_DECAY) # Decay eps over time to reduce random bids
        eps_threshold = EPS_END + (epsilon - EPS_END) * et
        reward=run_episode!(Q,df, episode, lr, eps_threshold, gamma, n_actions, lambda)
        
        rew += reward
        if rem(episode,1_000_000) == 1
            rew = 0
        elseif rem(episode,1_000_000) == 100001
            print(" P",i,"(",episode÷1_000_000, ") R=",rew/100000," eps=",eps_threshold) # Calc average reward of 100000 each miljon episode
            global rewardlog[i,1+(episode÷1_000_000)] = rew/100000
        end

        if episode >= (episodes-TESTEPISODES) # Only evaluate results from last episodes; when hopefully have learnt something..
            rewards += reward
        end    
    end
    return Q, rewards/TESTEPISODES
end

function save_Qdata(Q::Vector{Dict{Vector{Int8}, Vector{Float16}}})
  f = serialize("Qvaluefor1S1N.dat", Q)
  return f
end

function load_Qdata()
    Q = deserialize("Qvaluefor1S1N.dat")
    return Q
  end
  
function shape(state::Vector{Int8})
    return state[1]*1000+state[2]*100+state[3]*10+(13-state[3]-state[1]-state[2])
end  

function pb(bid::String) # Print N as NT because only N is to narrow in columns
    if bid[2] == 'N'
        return bid * "T"
    else
        return bid
    end    
end

function present_result(Q::Dict{Vector{Int8}, Vector{Float16}},i::Int64,io::IO,lr::Float64,rew::Float64,lambda)
    treexamples = []
    treebids = []
    print(io, "*** Try ",i, " Lr=",lr," R=",rew,"lambda=",lambda, "eps=",EPS_START, " - ",EPS_END)
    for p in range(11,22)
#        print(p," ")#f"{p:4}",end='')
    end
    printed = false
    println(io, " --- North 1st rebid ---")
    println(io,"     12   14   16   18   20")
    for s in range(5,7)
        for h in range(0,s)
            for d in range(0,min(13-s-h+1,s+1,6))
                c= 13 - s - h - d
                if c>=0 && c < s+1 && c < 7
                    for p in range(10,22)
                        state = convert(Vector{Int8},[s,h,d,p])
                        if haskey(Q,state)
                            if !printed
                                print(io, shape(state))
                            end
#                            print(io, " ",p,":")    # Print h.c.p.
                            printed = true
                            bb = argmax(Q[state])
                            if bb == 1 # N passes
                                print(io," P   ")
                            else
                                print(io, " ",pb(n2b(startbidn+bb-1))," ") # N can pass
                            end
                            for i=1:n_actions
#                                print(io, Q[state][i]," ") # Print Q-values for all actions
                            end
                            push!(treexamples,state) # To be filled with clubs and bid
                            #treebids.append(bids[selact])
                        end
                    end
                    if printed
                        println(io)
                        printed = false
                    end                        
                end
            end
        end
    end
    bids = 10
    println(io)
    println(io, "--- South's Responses ---")
    print(io, "- to: ") 
    for bid in range(1,bids)
        print(io, pb(n2b(bid+startbidn)),"     ")
    end
    println(io)    
    println(io,"h.c.p:8-9 10+ 8-9 10+ 8-9 10+")
    printed = false # Avoid blank lines
    for s in range(0,2) # Max 2 spades
        for h in range(0,6) # Max 6 H,D,C
            for d in range(0,6)#min(13-s-h+1,s+1))
                c= 13 - s - h - d
                if c>=0 && c < 7
                    #maxbid = minimum(10,(n_actions-1))
                    for bid in range(1,10) # Dont print when N Pass
                        for p in range(8,10,step = 2) # only even hcp. 11 is stored at 10
                            state = convert(Vector{Int8},[s,h,d,p,bid])
                            if haskey(Q,state)
                                if !printed
                                    if shape(state) < 100 # Pad major void shapes with 0
                                        print(io, "00")
                                    elseif shape(state) < 1000
                                        print(io, "0")
                                    end
                                    print(io, shape(state)," ")
                                    printed = true
                                end    
                                #printed = true
    #                                print(p,":")
                                bb = argmax(Q[state])
                                if bb == 1 # S passes
                                    print(io," P  ")
                                else
                                    print(io, " ",pb(n2b(startbidn+bb-1+bid)))
                                end    
                                for i=1:n_actions
    #                                    print(io, Q[state][i]," ") # Print Q-values for all actions
                                end
                                push!(treexamples,state) # To be filled with clubs and bid
                                #treebids.append(bids[selact])
                            else
                                print(io," -- ")    
                            end
                        end
                    end
                    println(io)
                end
#                if printed
#                    println(io)
                printed = false # New shape = new line
            end
        end
    end        
    #print(io, treexamples) 
end

function eval_sys(Q::Dict{Vector{Int8}, Vector{Float16}}, df::CSV.File, n_actions::Int32)
end

function test() # Check the bid-scoring (no X or XX test yet)
    for bid in range(0,40)
        println(bid,' ', n2b(bid), " ", b2n(n2b(bid))," 13:",score(n2b(bid),false,13)," ",score(n2b(bid),true,13))
    end

end

function main()
    # test()
    started = Dates.now()
    print("Started ",started," ! Loading data..")
    seed!(321) # For reproducibility
    fill_scoren()
    df = load_data()
    #reduce!(df)
    println(df[2][:sn],"spades on N. Loaded. Start training!")
    lr=[0.001, 0.001, 0.001, 0.001,  0.001, 0.001, 0.001, 0.001, 0.001, 0.001 ]#, 0.5, 0.1, 0.01, 0.001] # Maybe run each twice for statistics
    lambda=[0.5, 0.5 ,0.6 ,0.6 , 0.7 ,0.7 ,0.8 ,0.8, 0.9, 0.9 ]
    rews=zeros(length(lr))
    Q=Vector{Dict{Vector{Int8}, Vector{Float16}}}(undef,length(lr)) # Large Q(State,Value) "table" Small int&float to save space.
    Threads.@threads for i=1:length(lr) # Run multi-threaded in paralell
        Q[i] = Dict{Vector{Int8}, Vector{Float16}}() # Allocate sub-Q's
        q,r = train!(Q[i], df, EPISODES, lr[i], EPS_START, 0.95, n_actions,i,lambda[i]) # Gamma = 0.95; prefer shorter auctions, arriving later gives n * gamma * score
        println("Lr",lr[i]," reward=",r)
        rews[i] = r
    end    
    save_Qdata(Q) # All i's

    f = serialize("RLBRI_rewardlogfor1S1N.dat", rewardlog)
    print("Rewardlog=",rewardlog)

    open("RLBRI_result.txt", "w") do io
        for i=1:length(Q) # Don't present in paralell
            present_result(Q[i],i,io,lr[i],rews[i],lambda[i])
        end
    end
    finished = Dates.now()
    runtime = finished - started
    print("Ready after ", runtime, ". Episodes:", EPISODES) 
end
function main2() #Analyse results, only after previous run
    print("Loading Q-data..(3-4 mins?).. ")
    Q=load_Qdata()
    print("Loaded, creating resultfile..")
    #redirect_stdio(stdout="RLBRIstdout.txt", stderr="RLBRIstderr.txt") # Stopped working with Julia 1.7+
    open("RLBRI2_result.txt", "w") do io
        for i=1:length(Q) # Don't present in paralell
            present_result(Q[i],i,io,0.0,0.0,0.0)
        end
    end    
end

main() # change between main() = train and main2() = analyse

