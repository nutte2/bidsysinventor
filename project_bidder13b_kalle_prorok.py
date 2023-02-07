# This is a prototype programme made by Kalle Prorok 2023-02-07
# Version 2.0

# It tries to find best bid, given a situation in the card-game Bridge
#
# It presents what to bid with different shapes and strenghts.
# Only two bids are learnt in the current implementation
# Methods used are Deep Learning; algorithm:
# Neural net training with PyTorch (not Multiprocessing yet)
#
# Example Deals were generated with DealerV2, and it's built-in DDS.
# Runtime on an i7-8700K and Nvidia RTX A5000 ca 15 mins with 100 episodes
#
# Contact: kalle.prorok@gmail.com, Stockholm, Sweden

#import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import pandas as pd
import os
import os.path
#from collections import namedtuple, deque
#from itertools import count

from colorama import Fore, Back, Style
from sklearn import tree
from sklearn.tree import export_text
import torch
from torch import nn
from torch import optim
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
##from torch.nn import functional as F
#import gym
import torch.multiprocessing as mp

#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F

# --- Bridge part ---
# ♠♥♦♣

startbid="1N"

n_actions = 11 #len(bids)  # Number of bids to try
#learning_rate = 1e-4
onlyonebid = False # Only openers rebid
twobids = True # rebid + 2 nd response, if False; 3 bids
nopass = True #False # Opener not allowed to pass
#global_clc = 0.5


n_hand = 5 # shape hcp  # Shape+strength len(state), 6 will include LTC and seven also Controls
n_observations = n_hand*2+2*n_actions # with 2 hands and 3 new bids, 2 stored - input layer for ANN
batches = 32 # (64 don't work on 24 GB RTX A5000?)
epoches = 500

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
    if bid == "P":
        return 0

    if int(bid[0])>7:
        bid="7N"

    return 1+(int(bid[0])-1)*5+rank[bid[1]]

startbidn=b2n(startbid)

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

def imp(diff,topscore=True):
    if topscore:
        return diff
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
n_bids = 35

#        print(f"{self.env.finalbid} {result(self.env.ex,self.env.finalbid):4} {self.env.lastreward-impofs:5}({ self.env.par:5}) {n:17}{s:17} {ns} {es} {ss} {ws}" )


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

class Bidnet(nn.Module):
    def __init__(self):
#        super().__init__()
        super(Bidnet, self).__init__()
        self.A1 = nn.Linear(n_hand, 20)#,bias=False)
        self.A2 = nn.Linear(20, 20)#,bias=False)
        self.A3 = nn.Linear(20, n_actions)#,bias=False)

        self.B1 = nn.Linear(n_hand+n_actions, 30)
        self.B2 = nn.Linear(30, 30)
        self.B3 = nn.Linear(30, n_actions)

    def forward(self, x):
#        print(f"forward{x=}")
        x2 = x[:,0:n_hand] # Extract North's hand
#        print(f"forward {x2=}")
#        b1 = f.relu(self.A1(x2))
        b1 = self.A1(x2)
#        print(f"forwardb1a {b1=}")
        #b1 = f.relu(self.A2(b1))
        b1b = self.A2(b1)
#       print(f"b1b{b1=}")
#       b1 = f.relu(self.A3(b1))
        b1c = self.A3(b1b)
#        print(f"forward b1c{b1=}")
        x3 = x[:,n_hand:(n_hand+n_hand)] # Extract South's hand
#        print(f"forward{x3=}")
#        b1 = f.log_softmax(b1, dim=1)
        nbr = torch.argmax(b1c, dim=1) # Select only one bid
#        onehot=torch.zeros()

#        nb = torch.transpose(nb,1,0)
#        nb = torch.reshape(nbr,(batches,1))
#        print(f"forward {x3=}{nbr=}")
#        print(f"forward:{x2=} b1c:{b1=}{nbr=}")
#        if nbr > 0: N bids
#        print(f"forward {z=}")
#        nbs = torch.scatter(z,1,nb,1)
#        print(f"forward {nbs=}")
##        z = torch.zeros(batches, n_actions, device=device)
##        z[:, nbr] = 1  # Onehot for selected bid

#        b2in = torch.cat((x3, z), 1) # Input is South hand and onehot for N bid (incl pass)
        b2in = torch.cat((x3, b1c), 1)  # x[:,n_hand:(n_hand+n_hand)].expand(b1)
#        print(f"forward {b2in=}")
        b2 = f.relu(self.B1(b2in))
        b2 = f.relu(self.B2(b2))
        b2 = f.relu(self.B3(b2))
        # else: # N pass - don't work - different N bids in the batch
            #b2 = torch.zeros(batches,n_bids)
#        print(f"forw:{b1=}{b2=}{nbr=}")
        return b1,b2,nbr

    def bidA(self,n): # Doesn't work (yet), use forward with any S and skip its S bid
        b1 = f.relu(self.A1(n))
        b1 = f.relu(self.A2(b1))
        b1 = f.relu(self.A3(b1))
        return b1

"""        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
"""

bidnet = Bidnet().to(device)
print(bidnet)


TESTEXAMPLES = 2000
NUM_EPISODES = 100


steps_done = 0

episode_durations = []


# --- Main program stuff --
examples=0
df = pd.read_csv(r'res22k.txt') # Read Bridge hands incl resulting number of tricks to learn from
df_rows, df_columns = df.shape

df_rows = 20000 # during programming/test

##  maybe to be replaced with arr = np.loadtxt("sample_data.csv",delimiter=",", dtype=str)
#display(arr)
print(df) # Show some of them
print(df.info())

def checkexamples():

    print(df["pn"].describe(''))
    r,k = df.shape
    print("shape:",r,k)

best=np.zeros(df_rows,dtype=int)
#best=np.zeros(df_rows,dtype=float)
best_onehot=[]

def findbestcontract():
    """
    loop through all examples to find bid with highest score. Ties between i.e 4S and 4H are not handled yet.
    Save result in file for next run = much quicker
    Put the results into one-hot vector size 35 (all contracts) with a 1 at the proper place. More ones could be set later
    :return:
    """
    if os.path.isfile('bidder_bestcontract.csv'): # Already done
        globals()['best'] = loadtxt('bidder_bestcontract.csv', delimiter=',').astype(np.int64) # avoid local "best"
    else:
        for exp in range(1,df_rows):
            score=-5000
            for contract in range(5, 35):
                res = result(exp, n2b(contract))
                if res > score:
                    best[exp-1] = contract
                    be = contract
                    score = res
            n = list(df.iloc[exp, 4:9])
            s = list(df.iloc[exp, 11:16])
            ns = df.iloc[exp, 0]
            #es = df.iloc[exp, 1]
            ss = df.iloc[exp, 2]
            print(globals()['best'][exp-1],n2b(round(globals()['best'][exp-1])),n,s,ns,ss)
        savetxt('bidder_bestcontract.csv', best, delimiter=',')
    print(f"{best=}")
#    best_onehot = np.zeros((best.size, n_bids+1))#best.max() + 1))
#    best_onehot[np.arange(best.size), best] = 1
    globals()['best_onehot'] = np.eye(n_bids)[best] # Assign ones into right places.
    print(f"{best_onehot=}")

#trainingdata=np.zeros([df_rows,n_hand*2],dtype=int)
trainingdata=np.zeros([df_rows,n_hand*2],dtype=float) # Later to torch and to device

def createinputdata():
    """
    Scale examples like 5431, 12 into 0..1 range.
    Extract just useful info
    Save result in file for next run = much quicker
    :return:
    """
    if os.path.isfile('bidder_trainingdata.csv'):  # Already done
        globals()['trainingdata'] = loadtxt('bidder_trainingdata.csv', delimiter=',')#.astype(np.int64)  # avoid local var
    else:
        for exp in range(1,df_rows):
            hands = list(df.iloc[exp, 4:9])
            s = list(df.iloc[exp, 11:16])
            ns = df.iloc[exp, 0]
            # es = df.iloc[exp, 1]
            ss = df.iloc[exp, 2]
#            trainingdata[exp] = np.array(list(df.iloc[exp, 4:9] df.iloc[exp, 11:16])
            hands.extend(s)
            globals()['trainingdata'][exp-1] = np.array(hands) * np.array([0.1, 0.1, 0.1, 0.1, 0.05,0.1, 0.1, 0.1, 0.1, 0.05])
            if exp < 10:
                #print(hands,s,exp)
                print(globals()['trainingdata'][exp-1], hands, s, ns, ss)
        savetxt('bidder_trainingdata.csv', trainingdata, delimiter=',')
    print(trainingdata)


"""        
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
"""

# Support functions for Pytorch data loader set managment
class CustomBridgeDataset(Dataset):
    def __init__(self, annotations_file="", img_dir="", transform=None, target_transform=None):
        self.inited = True

    def __len__(self):
#        print(f"{len(best)=}")
        return len(best)

    def __getitem__(self, idx):
        #print(f"{idx=}{best[idx]=}{trainingdata[idx]=}")
#        return torch.Tensor(trainingdata[idx]), best_onehot[idx] #, torch.scatter(,n_bids)
        return trainingdata[idx], best_onehot[idx] #, torch.scatter(,n_bids)

training_datal = CustomBridgeDataset()
train_dataloader = DataLoader(training_datal, batch_size=batches, shuffle=True)

"""        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
"""
def loadmodel():
    bidnet = Bidnet()
    bidnet.load_state_dict(torch.load("bidder_model.pth"))

def train():
    t = time.localtime()
    print("Epoch Loss Hands Bestcontract N-bids S-bids. Started training",time.asctime(t) )

#    criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bidnet.parameters(), lr=0.0001)
 #   optimizer = optim.SGD(bidnet.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epoches):  # loop over the dataset multiple times
        te=time.time()
        print(f"{epoch=}")
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
#            print(f"forepoch:{i=}{data=}")
#            print(f"forepoch:{i=}")
            inputs, labels = data[0].to(device), data[1].to(device)
#            inputs, labels = data[0], data[1]

            # zero the parameter gradients
#            optimizer.zero_grad()
#            src = torch.zeros(1, n_bids)
#            print(f"fori1{labels=}")
#            labels = torch.torch.zeros(3, 5, dtype=src.dtype).scatter_(0,labels, n_bids,src)
#            labels = torch.torch.zeros(1, n_bids, dtype=src.dtype).scatter_(0,labels, 1)
#            print(f"fori2{labels=}")

            # forward + backward + optimize
            outputs = bidnet(inputs)
#            print(f"{outputs=}")
#            logits = bidnet(inputs)
#            logits = model(X)
#            outputs = nn.Softmax(dim=1)(logits)
 #           pred_probab = nn.Softmax(dim=1)(logits)
 #           y_pred = pred_probab.argmax(1)
#            labels = torch.scatter(0,labels,)
#            print(f"{labels=}")
           # op = outputs[0].cpu().detach()
            bidn = outputs[2]#.cpu().detach() # N selected best bid (not vector)
            bidns = bidn + startbidn
            bidns.to(device)
            if i % 100 == 1:
                #print(f"train:{outputs[0]=}{bidn=}")
                print(f"train:{bidns=}")

            ot = torch.zeros(batches,n_bids,device=device) # Generate a label-sized matrix
            for b in range(batches): # Fill with values at right places dependent of bidding
                ofset = startbidn + int(bidns[b].float()+0.5) # Emulate Round by adding 0.5 to int()
                if ofset == startbidn: # N Passes; final contract = start contract
                    ot[b,ofset] = 1
                else:
                    for sb in range(min(35-ofset,n_actions)): # Probably possible to vectorize..max 7 NT
                        val = outputs[1][b][sb].float()
    #                    print(f"{b=}{ofset=}{sb=}{ix=}")
                        ot[b, ofset + sb] = val #int(outputs[1,b, sb].float())
#            print(f"preloss {ot=}")
#            loss = f.nll_loss(ot, labels)
            loss = criterion(ot, labels)
#            print(f"forepoch, after criterion:{outputs=}{labels=}{loss=}")
#            print(f"forepoch:{epoch=}{i=}{loss=}")
            optimizer.zero_grad()
            loss.backward()
            #print(f"lossbackward ok")
            optimizer.step()
            #print(f"optimiz ok")

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'mb:[{epoch + 1}, {time.time()-te:.3} s {i + 1:5d}] loss: {running_loss / 200:.3f}')
                te = time.time()
                if True:#(epoch % 50 == 1) or (epoch < (epoches-10)): # print examples last 10 epochs
                    op=outputs[0].cpu().detach() # North bid vectors
                    ops=outputs[1].cpu().detach() # South bid vectors
#                op=f.normalize(op)
                    la=labels[0].cpu().detach() # Best Contracts to match with
                    bidA=op[0]
                    bidB=ops[0]
#                    print(f"{bidA},{bidB}")
                    inp=inputs[0].cpu().detach() # Input hands
                    inp = inp.numpy().tolist()
                    inpi=[]
                    for it in range(len(inp)):
                        if (it % 5) == 4:
                            inpi.append(int(inp[it]*20)) # shape normalized to /10, hcp scaled *0.05
                        else:
                            inpi.append(int(inp[it]*10)) # shape normalized to /10, hcp scaled *0.05
                    bidA = np.argmax(bidA)
                    if bidA > 0: # North bids other than Pass
                        bidB = np.argmax(bidB)
                    else:
                        bidB = 0

                    contr = np.argmax(la)

 #                   print(bidA,bidB)
#                    bidA = bidA.numpy()
 #                   bidA = bidA.float()
  #                  print(bidA,bidB)
                    bidA = int(bidA)
                    bidA += b2n(startbid)
                    bidB = int(bidB) + bidA
                    contr = int(contr)
                    contr = n2b(contr)
                    bidA = n2b(bidA)
                    bidB = n2b(bidB)
#                    with torch.no_grad():
#                        n = torch.Tensor([[5., 2., 3., 3., 14.]])
#                        n.to(device)
#                        bidA = bidnet.bidA(n)# torch.Tensor([5., 2., 3., 3., 14.]))#inputs[0]]))
                    #diff = op - la
#                    print(f"{epoch} {running_loss / 200:.3f}{inputs[0]}{la}:{contr}{op[0]} N:{bidA}{ops[0]} S:{bidB}")
                    if bidA == "P" or bidA == startbid:
                        #print("N passes:")
                        print(f"{epoch} {running_loss / 200:.3f} {inpi} {contr} N:Pass at {startbid}")
                    elif bidB == bidA:
                        print(f"{epoch} {running_loss / 200:.3f} {inpi} {contr} N:{bidA} S:Pass")
                    else:
                        print(f"{epoch} {running_loss / 200:.3f} {inpi} {contr} N:{bidA} S:{bidB}")
                    torch.save(bidnet.state_dict(), "bidder_model_temp.pth")
                running_loss = 0.0

    torch.save(bidnet.state_dict(), "bidder_model.pth")
    print("Saved PyTorch Model State to bidder_model.pth")
    t2 = time.localtime()
    print("Finished training",time.asctime(t2) )

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
##        bidsnames=[b for b in bids]
##        tree.plot_tree(clf,feature_names=["♠", "♥", "♦", "♣", "P", "B4", "B5"], class_names=bidsnames,precision=2)
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


def showresult(last=False):
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
                        state_ = np.zeros(n_observations)
                        hand = [s, h, d, c, p]
                        state_[0:5] = np.array(hand)  # North hand always start
                        state = torch.from_numpy(state_).float()
                        logits, value = 0,0 #MasterNode(state)
                        action_dist = torch.distributions.Categorical(logits=logits)
                        action = action_dist.sample()
                        actionno = action.detach().numpy()
                        actiondist2 =  action_dist.logits
                        if nopass:
                            actionno += 1
                        bid = n2b(actionno+b2n(startbid))
                        ##print(f"showresult:{actiondist2=} {actionno=}{bid=}")
                        ##state = torch.tensor(hand, dtype=torch.float32, device=device).unsqueeze(0)
#                        selact = policy_net(state).max(1)[1].view(1, 1)
                        p=percentage(hand)
                        if p != " " and p != ".":
                            cprint(f"{bid:2} ")
                            print(p,end=' ')
                            treexamples.append(hand)
                            treebids.append(bid)
                        else:
                            print ("    ",end="")
                    print('')
    #showtree(treexamples,treebids,last)

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
#MasterNode = ActorCritic()
#MasterNode.share_memory()
processes = []
params = {
#    'epochs': 5,
#    'n_workers': 2,
    'epochs': 5002,
    'n_workers': 10,
}

def testfcns():
    for bno in range(0,36):
        bid=n2b(bno)
        bno2=b2n(bid)
        print(bid,bno)
        if bno != bno2:
            print("Error",bid)

def testAB():
#dataiter = iter(testloader)
#images, labels = next(dataiter)
    pass

# print images
#imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

if __name__ == '__main__':  # adding this for process safety
    #testfcns()
    findbestcontract()
    createinputdata()
    print(f"{best_onehot}")
    trainingdata = torch.Tensor(trainingdata)
#    trainingdata.to(device)
    best_onehot = torch.Tensor(best_onehot)
#    best_onehot.to(device)
    train()
    #testAB()
    loadmodel()

"""    
    for global_clc in [0.1]:
        for learning_rate in [0.00001]:
            print(f"{global_clc=}")
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
            #torch.save(MasterNode, "masterDA2C_1bid.pt")
            test()
        #    test()
            showresult(True)
        #    showresult(True)

"""
# End of program

