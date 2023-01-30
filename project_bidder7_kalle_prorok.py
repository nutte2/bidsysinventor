# This is a prototype programme made by Kalle Prorok 2023-01-26
# Version 2.0

# It tries to find best bid, given a situation in the card-game Bridge
#
# It presents what to bid with different shapes and strenghts.
# Only two bids are learnt in the current implementation
# Methods used are Deep Reinforcement Learning, algorithm:
# Distributed Advantage Actor-Critic (DA2C) with PyTorch+Multiprocessing
#
# Example Deals were generated with DealerV2, and it's built-in DDS.
# Runtime on an i7-8700K and Nvidia RTX A5000 ca 15 mins with 50_000 episodes
#
# Contact: kalle.prorok@gmail.com, Stockholm, SwedenS

#import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from collections import namedtuple, deque
#from itertools import count

from colorama import Fore, Back, Style
from sklearn import tree
from sklearn.tree import export_text
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
#import gym
import torch.multiprocessing as mp

#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F

# --- Bridge part ---
# ♠♥♦♣
# The bids to try to find which is best in given situations
bids = ["P", "2♣", "2♦", "2♥", "2♠", "2N", "3♣", "3♦", "3♥", "3♠","3N", "4♥", "4♠", "5♣", "5♦", "6♠", "6N"]

startbid="1N"

n_actions = 12 #len(bids)  # Number of bids to try
n_observations = 7 # shape hcp prevbid1 prevbid2 - 5 # Shape+strength len(state)

suits=["♣", "♦", "♥", "♠", "N"]
rank={"♣":0, "♦":1, "♥":2, "♠":3, "N":4 }

def n2b(num):
    """
Convert a number to a bid-string. 0 gives Pass.  
    :param num: 
    :return: 
    """
    if num > 35:
        num = 35
    if num == 0:
        return "P"
    else:
        level = 1+(num-1)//5
        suit = suits[(num-1)%5]
        return str(level) + suit


def b2n(bid):
    """
Convert a str bid into a ranking number. P=0, 1C=1, 2C = 6..
    :param bid: 
    :return: 
    """
    if int(bid[0])>7:
        bid="7N"
    if bid == "P":
        return 0
    else:
        return 1+(int(bid[0])-1)*5+rank[bid[1]]

def result(ex, bidst):
    """Calculates the score on a given example+bid, negative if not made
    :param ex: number of the example line 
    :param bidst: selected bid string on this example
    :return: number of resulting Bridge points on this example given the bid
    """
    ret = 0
##    bid = bid[0][0]
#    print(f"Exno {ex} Bid:{bid}")
    dstate = list(df.iloc[ex, 18:23])
    if bidst == "P":
        bidst = startbid
##    else:
##        bidst = bids[bid]

#    print(dstate[5],int(bidst[0]))

    if bidst[1] == "N":  # No Trump
        if dstate[4] >= int(bidst[0])+6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 990 + (dstate[4] - int(bidst[0]) - 6) * 30
            elif int(bidst[0]) >= 3:  # Bid game level
                ret = 400 + (dstate[4]-int(bidst[0])-6)*30
            else:
                ret = 60 + (dstate[4]-6)*30
        else:
            ret = 50*(dstate[4]-int(bidst[0])-6)  # Failed
    elif bidst[1] == "♠":  # Spades
        if dstate[3] >= int(bidst[0])+6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 980 + (dstate[3] - int(bidst[0]) - 6) * 30
            elif int(bidst[0]) >= 4:  # Bid game level
                ret = 420 + (dstate[3]-int(bidst[0])-6)*30
            else:
                ret = 50 + (dstate[3]-6)*30
        else:
            ret = 50*(dstate[3]-int(bidst[0])-6)  # Failed
    elif bidst[1] == "♥":  # Hearts
        if dstate[2] >= int(bidst[0])+6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 980 + (dstate[2] - int(bidst[0]) - 6) * 30
            elif int(bidst[0]) >= 4:  # Bid game level
                ret = 420 + (dstate[2]-int(bidst[0])-6)*30
            else:
                ret = 50 + (dstate[2]-6)*30
        else:
            ret = 50*(dstate[2]-int(bidst[0])-6)  # Failed
    elif bidst[1] == "♦":
        if dstate[1] >= int(bidst[0]) + 6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 920 + (dstate[1] - int(bidst[0]) - 6) * 20
            elif int(bidst[0]) >= 5:  # Bid game level
                ret = 400 + (dstate[1] - int(bidst[0]) - 6) * 20
            else:
                ret = 50 + (dstate[1] - 6) * 20
        else:
            ret = 50 * (dstate[1] - int(bidst[0]) - 6)  # Failed
    elif bidst[1] == "♣":  #:
        if dstate[0] >= int(bidst[0]) + 6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 920 + (dstate[0] - int(bidst[0]) - 6) * 20
            elif int(bidst[0]) >= 5:  # Bid game level
                ret = 400 + (dstate[0] - int(bidst[0]) - 6) * 20
            else:
                ret = 50 + (dstate[0] - 6) * 20
        else:
            ret = 50 * (dstate[0] - int(bidst[0]) - 6)  # Failed
#    print(ret)#,end=' ')
    return ret

def imp(diff):
    if diff < 0:
        sgn=-1
        diff = -diff
    else:
        sgn = 1

    if diff<20:
        return 0
    elif diff < 50:
        return sgn
    elif diff < 90:
        return sgn * 2
    elif diff < 130:
        return sgn * 3
    elif diff < 170:
        return sgn * 4
    elif diff < 220:
        return sgn * 5
    elif diff < 270:
        return sgn * 6
    elif diff < 320:
        return sgn * 7
    elif diff < 370:
        return sgn * 8
    elif diff < 430:
        return sgn * 9
    elif diff < 500:
        return sgn * 10
    elif diff < 600:
        return sgn * 11
    elif diff < 750:
        return sgn * 12
    elif diff < 900:
        return sgn * 13
    elif diff < 1100:
        return sgn * 14
    elif diff < 1300:
        return sgn * 15
    elif diff < 1500:
        return sgn * 16
    elif diff < 1750:
        return sgn * 17
    elif diff < 2000:
        return sgn * 18
    elif diff < 2250:
        return sgn * 19
    elif diff < 2500:
        return sgn * 20
    elif diff < 3000:
        return sgn * 21
    elif diff < 3500:
        return sgn * 22
    elif diff < 4000:
        return sgn * 23
    else:
        return sgn * 24

impofs = 20 # offset to be added to imp-results to get positive scores during learning
learning_rate = 1e-4
onlyonebid = False # Only openers rebid
twobids = False # rebid + 2 nd response, if False; 3 bids
nopass = True # Opener not allowed to pass

class bgymEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def start(self):
        r, _ = df.shape
        training = r - TESTEXAMPLES
        self.ex = random.randrange(1, training)
        self.state = np.array(list(df.iloc[self.ex,4:11])) # North hand always start
        self.b1 = self.b2 = -1
        self.state[5] = self.b1
        self.state[6] = self.b2
        self.finalbid = startbid
        self.par = list(df.iloc[self.ex,38:39])[0]

    def __init__(self, render_mode=None, size=5):
        self.start()

    def reset(self):
        self.start()

    def step(self,action):
        #print(f"step:action:",action)
        if self.state[5] == -1: # first bid b1
            if nopass:
                action += 1
            if action == 0: # pass
                reward = imp(result(self.ex,self.finalbid) - self.par) + impofs
                observation = self.state
                terminated = True
                truncated = False
            else:
                newbid = min(b2n(self.finalbid) + action,33) #33 = 7N
                self.finalbid = n2b(newbid)
                self.state = np.array(list(df.iloc[self.ex,11:18])) # souths hand
                self.state[5] = newbid
                self.state[6] = -1
                if onlyonebid:
                    terminated = True
                    reward = imp(result(self.ex,self.finalbid) - self.par) + impofs
                else:
                    terminated = False
                    reward = 0
                truncated = False
                observation = self.state
        elif self.state[6] == -1: # Second bid
            if action == 0: # pass
                reward = imp(result(self.ex,self.finalbid) - self.par) + impofs
                observation = self.state
                terminated = True
                truncated = False
            else:
                newbid = min(b2n(self.finalbid) + action,33)
                oldb1 = self.state[5]
                self.finalbid = n2b(newbid)
                self.state = np.array(list(df.iloc[self.ex,4:11])) # norths hand
                self.state[5] = oldb1
                self.state[6] = newbid
                if twobids:
                    reward = imp(result(self.ex, self.finalbid) - self.par) + impofs
                    terminated = True
                else:
                    terminated = False
                    reward = 0
                truncated = False
                observation = self.state
        else: # Third bid
            if action == 0: # pass
                reward = imp(result(self.ex,self.finalbid) - self.par) + impofs
                observation = self.state
                terminated = True
                truncated = False
            else:
                newbid = min(b2n(self.finalbid) + action,35)
                self.finalbid = n2b(newbid)
                oldb1 = self.state[5]
                oldb2 = self.state[6]

                reward = imp(result(self.ex,self.finalbid) - self.par) + impofs
                self.state = np.array(list(df.iloc[self.ex,4:11])) # norths hand
                self.state[5] = oldb1
                self.state[6] = oldb2
                terminated = True
                truncated = True
                observation = self.state
#        if terminated:
#            print(f"Bid:{self.finalbid} Reward {reward}")
        self.lastreward = reward
        #print(f"{observation}{reward=}")
        return observation, reward, terminated, truncated#, info

class Bgym:
    def __init__(self, render_mode=None, size=5):
        self.made = False

    def make(self, name):
        self.name = name
        self.state = 0
        self.env = bgymEnv()
        return self

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
#        print('reset')

    def render(self):
        n=list(df.iloc[self.env.ex,4:9])
        s=list(df.iloc[self.env.ex,11:16])
        ns=df.iloc[self.env.ex,0]
        es=df.iloc[self.env.ex,1]
        ss=df.iloc[self.env.ex,2]
        ws=df.iloc[self.env.ex,3]
        print(self.env.finalbid, result(self.env.ex, self.env.finalbid), self.env.lastreward-impofs, "(", self.env.par, ")", n, s, ns, es, ss, ws )

##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##print(device)

"""        self.l1 = nn.Linear(n_observations,50)  #4
        self.l2 = nn.Linear(50,150)
        #        self.l2b = nn.Linear(150,150)
        self.actor_lin1 = nn.Linear(150,n_actions) #2
        self.l3 = nn.Linear(150,25)
        self.critic_lin1 = nn.Linear(25,1)
"""

# --- DA2C part
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(n_observations,50)  #4
        self.l2 = nn.Linear(50,50)
#        self.l2b = nn.Linear(100,75)
        self.actor_lin1 = nn.Linear(50,n_actions) #2
        self.l3 = nn.Linear(50,50)
        self.critic_lin1 = nn.Linear(50,1)

    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
#        y = F.relu(self.l2b(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic

def worker(t, worker_model, counter, params):
    bgym = Bgym()
    worker_env = bgym.make("BridgeInventor-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=learning_rate, params=worker_model.parameters()) # lr 1e-4
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss,critic_loss,eplen = update_params(worker_opt, values, logprobs, rewards)
        counter.value = counter.value + 1
        if i % 10000 == 0:
            print(f"{i}:{actor_loss=},{critic_loss=}")

def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    while (done == False):
        j+=1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, r, done, _ = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = r
            #print(f"r_e {r=}")
            worker_env.reset()
        else:
            reward = 0
        rewards.append(reward)
    return values, logprobs, rewards


def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -1*logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    loss = actor_loss.sum() + clc*critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)



def test():
    bgym = Bgym()
    env = bgym.make("BridgeInventor-v1")
    env.reset()
    imps=0
    contracts = 0
    print("1♠-Bidding Contract Score Impdiff (Parscore) N[shape,hcp] S[shape,hpc]  N-cards E-cards S-cards W-cards")
    for i in range(100):
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        logits, value = MasterNode(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        print(env.env.finalbid,end=' ')
        state2, reward, done, info = env.step(action.detach().numpy())
        if done:
            #print()
            #print(state2,reward)
            imps += reward
            contracts += 1
            env.render()
            env.reset()
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
    print(f"Average imps {imps/contracts-impofs:.2}")
# --- End of DA2C part


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 8192 #2048 #512 #128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.005
#EPS_DECAY = 1000
#EPS_DECAY = 20000
EPS_DECAY = 40000
TAU = 0.005
LR = 1e-4
#LR = 10e-4

TESTEXAMPLES = 2000
NUM_EPISODES = 100


steps_done = 0

episode_durations = []


# --- Main program stuff --
examples=0
df = pd.read_csv(r'res22k.txt') # Read Bridge hands incl resulting number of tricks to learn from

##  maybe to be replaced with arr = np.loadtxt("sample_data.csv",delimiter=",", dtype=str)
#display(arr)
print(df) # Show some of them
print(df.info())
def checkexamples():

    print(df["pn"].describe(''))
    r,k = df.shape
    print("shape:",r,k)

"""
def run():
#    """ """    :return: nothing but plots result and updates policy-models
    """ """
    reward = 0
    runscore=np.arange(NUM_EPISODES)

    if torch.cuda.is_available():
        num_episodes = NUM_EPISODES
    else:
        num_episodes = 50
    eps = EPS_START
    r, _ = df.shape
    training = r -  TESTEXAMPLES
    for i_episode in range(num_episodes):
        if i_episode % 10000 == 10:
            print(num_episodes-i_episode,eps,reward[0]) # Show count-down as a sign of life
            showresult(False)
            evalmodel()
        ex=random.randrange(1,training) # Example number, save last 1000 for valid/test
        dstate=list(df.iloc[ex,4:11]) # Start with different examples, added 2 for prev_bid space North, rebidder
        slstate=list(df.iloc[ex,11:18]) # South Hand
#        print(dstate)
        state=dstate[0:7] # Shape, strength + space for prevbids
        sstate=sstate[0:7]

#        print(state)
        state[5] = -1 # no prevbids
        state[6] = -1

        sstate[5] = -1  # no prevbids
        sstate[6] = -1
#        print(state)

        par= list(df.iloc[ex,38:39])[0]
#        print("par=",par)
#        print(state)
        tstate = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tsstate = torch.tensor(sstate, dtype=torch.float32, device=device).unsqueeze(0)

        prevact= -1#False
        for t in count():
            action,eps = select_action(tstate)
            observation = [0, 0, 0, 0, 0, -1, -1]
            if prevact == -1: # False:
                reward = imp(result(ex, action) - par)
                prevact = action
                terminated = False
                truncated = False
            elif prevact == 0: #pass
                reward = imp(result(ex, prevact) - par)
                runscore[i_episode] = reward
                terminated=True
                truncated=True
            else # not passed, try to find continuing bid
                reward = imp(result(ex, action) - par)
                prevact = action

                reward = imp(result(ex, prevact) - par)
                runscore[i_episode] = reward
                terminated=True
                truncated=True

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(reward)#t + 1)
#                plot_durations()
                break

    print('Complete')
#    torch.save(policy_net.state_dict(), "policy")
#    torch.save(target_net.state_dict(), "target")
#    torch.save(policy_net, "policy.pt")
#    torch.save(target_net, "target.pt")

    print("All Runs Imp = - par :Mean:",np.mean(runscore)," Sum:", np.sum(runscore), "Std:", np.std(runscore), "Median:", np.median(runscore))
    counts, bins = np.histogram(runscore, bins=12)
    plt.figure(4, figsize=(40, 32))
    plt.stairs(counts, bins)
    plt.show()
    plt.figure(5, figsize=(40, 32))

    #t = np.arange(0.0, 2.0, 0.01)
    #s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(range(num_episodes), runscore)

    ax.set(xlabel='episode)', ylabel='score',
           title='Performance learnt')
    ax.grid()

    fig.savefig("episodesscore.png")
    plt.show()

    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

def loadmodel():
    policy_net = torch.load("policy.pt")
    policy_net.eval()
    target_net = torch.load("target.pt")
    target_net.eval()

    #device = torch.device("cuda")
    #model = TheModelClass(*args, **kwargs)
    #model.load_state_dict(torch.load(PATH))
    #model.to(device)
"""

def showtree(x,y,last):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(x,y)
#    print(X,y)
    if last:
#        plt.rcParams['figure.figsize'] = [80, 64]  # Inches in 100 dpi; 1/3 in 300 dpi
        plt.figure(2,figsize=(40,32))
        bidsnames=[b for b in bids]
        tree.plot_tree(clf,feature_names=["♠", "♥", "♦", "♣", "P", "B4", "B5"], class_names=bidsnames,precision=2)
        plt.pause(0.001)
    r = export_text(clf, feature_names=["♠", "♥", "♦", "♣", "P", "B4", "B5"])
    print()
    print(r)


def cprint(bid):
    bg=Back.BLACK
    if bid[0] == "P":
        print(Fore.GREEN+bid[0]+" "+Style.RESET_ALL,end='')
    elif bid[1] == "♣":
        print(bid[0]+Fore.GREEN + bid[1] + Style.RESET_ALL, end='')
    elif bid[1] == "♦":
        print(bid[0]+Fore.YELLOW + bid[1] + Style.RESET_ALL, end='')
    elif bid[1] == "♥":
        print(bid[0]+Fore.RED + bid[1] + Style.RESET_ALL, end='')
    elif bid[1] == "♠":
        print(bid[0]+Fore.BLUE + bid[1] + Style.RESET_ALL, end='')
    elif bid[1] == "N":
        print(bid[0]+Fore.WHITE + bid[1] + Style.RESET_ALL, end='')
    else:
        print(bid, end='')
    #if bid[0] != "P": print(' ',end='')


"""
U+2581 	▁ 	Lower one eighth block
U+2582 	▂ 	Lower one quarter block
U+2583 	▃ 	Lower three eighths block
U+2584 	▄ 	Lower half block
U+2585 	▅ 	Lower five eighths block
U+2586 	▆ 	Lower three quarters block
U+2587 	▇ 	Lower seven eighths block
U+2588 	█ 	Full block 
"""
blockchar=[" ",".","▁","▂","▃","▄","▆","▇","█","+", ]

def percentage(hand):
    r, k = df.shape
#    print(df.columns.values.tolist())
    no = len(df[(df['sn'] == hand[0]) & (df['hn'] == hand[1]) &
        (df['dn'] == hand[2]) & (df['cn'] == hand[3]) & (df['pn'] == hand[4])])
    dpromille = 10000*no/r # decipromille; if r=20000 two examples are needed for a dot(.)
    if dpromille > 9:
        dpromille = 9
    return blockchar[int(dpromille)]


def showresult(last):
    """ Show a table with all shapes and strenghts with recommended bids
    :return: 
    """
    treexamples = []
    treebids = []
    print("   ",end='')
    for p in range(11,22):
        print(f"{p:4}",end='')
    print("")
    for s in range(5,8):
        for h in range(0,s+1):
            for d in range(0,min(13-s-h+1,s+1)):
                c= 13 - s - h - d
                if c>=0 and c < s+1:
#                    shape=

                    shape = str(s)
                    if h==0:
                        shape=shape+"-"
                    else:
                        shape = shape + str(h)

                    if d==0:
                        shape=shape+"-"
                    else:
                        shape = shape + str(d)
                    if c==0:
                        shape=shape+"-"
                    else:
                        shape = shape + str(c)

                    print(shape,end=' '),
                    for p in range(11,22):
                        hand=[s, h, d, c, p, -1, -1 ]
                        ##state = torch.tensor(hand, dtype=torch.float32, device=device).unsqueeze(0)
#                        selact = policy_net(state).max(1)[1].view(1, 1)
                        selact=0
                        p=percentage(hand)
                        if p != " " and p != ".":
                            cprint(f"{bids[selact]:2} ")
                            print(p,end=' ')
                            treexamples.append(hand)
                            treebids.append(bids[selact])
                        else:
                            print ("    ",end="")
                    print('')
    showtree(treexamples,treebids,last)

def evalmodel():
    r, k = df.shape
    score=np.arange(TESTEXAMPLES)

    for i in range(r-TESTEXAMPLES,r):
        hand = list(df.iloc[i,4:11])[0:7] # [s, h, d, c, p, prebid1, prebid2]
        hand[5] = -1
        hand[6] = -1
        ##state = torch.tensor(hand, dtype=torch.float32, device=device).unsqueeze(0)
        action = 0 # policy_net(state).max(1)[1].view(1, 1)
        par = list(df.iloc[i, 38:39])[0]
        score[i-r+TESTEXAMPLES] = imp(result(i, n2b(action)) - par)
    print("Eval imp (-par):Mean:",np.mean(score), "Stddev:", np.std(score), "Median:", np.median(score))

    counts, bins = np.histogram(score,bins=12)
    plt.figure(3, figsize=(40, 32))
    plt.stairs(counts, bins)
    plt.show()


"""
if __name__ == '__main__':
    random.seed(42)
    checkexamples()
#    run()
    #loadmodel() # does not work yet
    showresult(True)
#    evalmodel()
"""
counter = mp.Value('i', 0)

print("cpu_count mp:", mp.cpu_count())
MasterNode = ActorCritic()
MasterNode.share_memory()
processes = []
params = {
#    'epochs': 30,
#    'n_workers': 2,
    'epochs': 200000,
    'n_workers': 10,
}


if __name__ == '__main__':  # adding this for process safety
    for learning_rate in [1e-6]:#4, 1e-5, 1e-6]:#, 1e-5]:
        print(f"{learning_rate=}")
        for i in range(params['n_workers']):
            p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
        print(counter.value, processes[1].exitcode)
        test()


# End of program

