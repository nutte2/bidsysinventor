# This is a prototype programme made by Kalle Prorok 2023-01-05
# Version 1.0

# It tries to find best bid, given a situation in the card-game Bridge
#
# It presents what to bid with different shapes and strenghts.
# Only one bid is learnt in the current implementation
# Methods used are Deep Reinforcement Learning, algorithm: DQN from PyTorch
#
# Example Deals were generated with DealerV2, and it's built-in DDS.
# Runtime on an i7-8700K and Nvidia RTX A5000 ca 15 mins with 50_000 episodes
#
# Contact: kalle.prorok@gmail.com, Stockholm, Sweden

import math
import random
# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import namedtuple, deque
from itertools import count

from colorama import Fore, Back, Style
from sklearn import tree
from sklearn.tree import export_text
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Bridge part ---
# ♠♥♦♣
# The bids to try to find which is best in given situations
bids = ["P", "2♣", "2♦", "2♥", "2♠", "3N", "4♥", "4♠", "5♣", "5♦", "6♠", "6N"]

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

def result(ex, bid):
    """Calculates the score on a given example+bid, negative if not made
    :param ex: number of the example line 
    :param bid: selected bid on this example
    :return: number of resulting Bridge points on this example given the bid
    """
    ret = 0
    bid = bid[0][0]
#    print(f"Exno {ex} Bid:{bid}")
    dstate = list(df.iloc[ex, 18:23])
    if bids[bid] == "P":
        bidst = "1N"
    else:
        bidst = bids[bid]

#    print(dstate[5],int(bidst[0]))

    if bidst[1] == "N":  # No Trump
        if dstate[4] >= int(bidst[0])+6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 990 + (dstate[4] - int(bidst[0]) - 6) * 30
            elif int(bidst[0]) >= 3:  # Bid game level
                ret = 400 + (dstate[4]-int(bidst[0])-6)*30
            else:
                ret = 90 + (dstate[4]-int(bidst[0])-6)*30
        else:
            ret = 50*(dstate[4]-int(bidst[0])-6)  # Failed
    elif bidst[1] == "♠":  # Spades
        if dstate[3] >= int(bidst[0])+6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 980 + (dstate[3] - int(bidst[0]) - 6) * 30
            elif int(bidst[0]) >= 4:  # Bid game level
                ret = 420 + (dstate[3]-int(bidst[0])-6)*30
            else:
                ret = 80 + (dstate[3]-int(bidst[0])-6)*30
        else:
            ret = 50*(dstate[3]-int(bidst[0])-6)  # Failed
    elif bidst[1] == "♥":  # Hearts
        if dstate[2] >= int(bidst[0])+6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 980 + (dstate[2] - int(bidst[0]) - 6) * 30
            elif int(bidst[0]) >= 4:  # Bid game level
                ret = 420 + (dstate[2]-int(bidst[0])-6)*30
            else:
                ret = 80 + (dstate[2]-int(bidst[0])-6)*30
        else:
            ret = 50*(dstate[2]-int(bidst[0])-6)  # Failed
    elif bidst[1] == "♦":
        if dstate[1] >= int(bidst[0]) + 6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 920 + (dstate[1] - int(bidst[0]) - 6) * 20
            elif int(bidst[0]) >= 5:  # Bid game level
                ret = 400 + (dstate[1] - int(bidst[0]) - 6) * 20
            else:
                ret = 70 + (dstate[1] - int(bidst[0]) - 6) * 20
        else:
            ret = 50 * (dstate[1] - int(bidst[0]) - 6)  # Failed
    elif bidst[1] == "♣":  #:
        if dstate[0] >= int(bidst[0]) + 6:  # Made the contract, taken is at least bid
            if int(bidst[0]) >= 6:  # Bid small slam level
                ret = 920 + (dstate[0] - int(bidst[0]) - 6) * 20
            elif int(bidst[0]) >= 5:  # Bid game level
                ret = 400 + (dstate[0] - int(bidst[0]) - 6) * 20
            else:
                ret = 70 + (dstate[0] - int(bidst[0]) - 6) * 20
        else:
            ret = 50 * (dstate[0] - int(bidst[0]) - 6)  # Failed
#    print(ret)#,end=' ')
    return ret

# --- DQN part ---
#  Based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?highlight=dqn

# Check if CUDA (Nvidia GPU) is available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Named tuple for the DQN
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Used for retrying old experiences
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q-learning neural Net class
    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

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
NUM_EPISODES = 100000
n_actions = len(bids)  # Number of bids to try
n_observations = 5 # Shape+strength len(state)


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    """ Selects an action when in state, either random action or "best" action
    :param state: situation, here Bridge hand shape nand strength 
    :return: either random action or "best" action 
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
#        print('q')
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            selact = policy_net(state).max(1)[1].view(1, 1)
 #           print(selact)
            return selact, eps_threshold
    else:
#        print('e')
        selacte =  torch.tensor([[random.randrange(0, n_actions)]], device=device, dtype=torch.long)
#        print(selacte)
        return selacte, eps_threshold # torch.tensor([[random.randrange(0,n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    """ Plot performance
    :param show_result: used for showing final Result, otherwise runnig score 
    :return: nothing
    """
#    plt.rcParams['figure.figsize'] = [20,16] # Inches in 100 dpi; 1/3 in 300 dpi
    plt.figure(1,figsize=(20,16))
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
#    if is_ipython:
#        if not show_result:
#            display.display(plt.gcf())
#            display.clear_output(wait=True)
#        else:
#            display.display(plt.gcf())

def optimize_model():
    """ Improve the DQN model based on RL experiences
    :return: 
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# --- Main program stuff --
examples=0
df = pd.read_csv(r'res22k.txt') # Read Bridge hands incl resulting number of tricks to learn from
print(df) # Show some of them
print(df.info())
def checkexamples():

    print(df["pn"].describe(''))
    r,k = df.shape
    print("shape:",r,k)


def run():
    """ Run the training. Improvement: send num_episodes as parameter
    :return: nothing but plots result and updates policy-models
    """
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
        dstate=list(df.iloc[ex,4:9]) # Start with different examples

#        print(dstate)
        state=dstate[0:5] # Shape, strength
        par= list(df.iloc[ex,38:39])[0]
#        print("par=",par)
#        print(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        prevact= -1#False
        for t in count():
            action,eps = select_action(state)
            observation = [0, 0, 0, 0, 0]
            if prevact == -1: # False:
                reward = result(ex, action) - par
                prevact = action
                terminated = False
                truncated = False
            else:
                reward = result(ex, prevact) - par
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
    torch.save(policy_net, "policy.pt")
    torch.save(target_net, "target.pt")

    print("Run= - par :Mean:",np.mean(runscore)," Sum:", np.sum(runscore), "Std:", np.std(runscore), "Median:", np.median(runscore))
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

def showtree(x,y,last):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(x,y)
#    print(X,y)
    if last:
#        plt.rcParams['figure.figsize'] = [80, 64]  # Inches in 100 dpi; 1/3 in 300 dpi
        plt.figure(2,figsize=(40,32))
        bidsnames=[b for b in bids]
        tree.plot_tree(clf,feature_names=["♠","♥","♦","♣","P"],class_names=bidsnames)
        plt.pause(0.001)
    r = export_text(clf, feature_names=["♠","♥","♦","♣","P"])
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
                        hand=[s, h, d, c, p ]
                        state = torch.tensor(hand, dtype=torch.float32, device=device).unsqueeze(0)
                        selact = policy_net(state).max(1)[1].view(1, 1)
                        cprint(f"{bids[selact]:2} ")
                        print(percentage(hand),end=' ')
                        treexamples.append(hand)
                        treebids.append(bids[selact])
                    print('')
    showtree(treexamples,treebids,last)

def evalmodel():
    r, k = df.shape
    score=np.arange(TESTEXAMPLES)

    for i in range(r-TESTEXAMPLES,r):
        hand = list(df.iloc[i,4:9])[0:5] # [s, h, d, c, p]
        state = torch.tensor(hand, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy_net(state).max(1)[1].view(1, 1)

        score[i-r+TESTEXAMPLES] = result(i, action)
    print("Eval (no par):Mean:",np.mean(score)," Sum:", np.sum(score), "Std:", np.std(score), "Median:", np.median(score))

    counts, bins = np.histogram(score,bins=12)
    plt.figure(3, figsize=(40, 32))
    plt.stairs(counts, bins)
    plt.show()



if __name__ == '__main__':
    random.seed(42)
    checkexamples()
    run()
    #loadmodel() # does not work yet
    showresult(True)
    evalmodel()

# End of program

