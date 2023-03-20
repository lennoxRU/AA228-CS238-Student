import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random

#start time
st = time.time()

class POMDP:
    def __init__(self, S, A, T, R, O, eps, U, P):
        self.S = S      # state space
        self.A = A      # action space
        self.T = T      # transition matrix
        self.R = R      # reward sum r(s,a)
        self.O = O      # observation model
        self.eps = eps  # discount
        self.U = U      # value function
        self.P = P      # current policy


# simulate a game with one student once
def simulate(model):
    
    start = random.randint(1, len(model.S))

class ParticleFilter:
    def __init__(self, states):
        self.states = states      # vector of state samples

# belief updater for particle filter, which updates a vector of states 
# representing the belief based on the agents action a and observation o
def update(b, model, a, o):
    states = [random.choice(model.T(s,a)) for s in b.states]
    weights = [model.O(a, sn, o) for sn in states]
    return ParticleFilter([random.choice(states, p=weights) for i in range(len(states))])

def lookahead(model, s, a):
    S, U, T, R, eps, P = model.S, model.U, model.T, model.R, model.eps, model.P
    ret = R[s-1,a-1] + eps*(max(T[s-1,a-1,sn-1]*U[sn-1] for sn in model.S))
    #print ("for (s,a) = ", s,",", a, "r= ",ret)
    return ret

def backup(model, s):
    maxvalues= [lookahead(model, s, a) for a in model.A]
    print("s:",s)
    print("maxvalues: ", maxvalues)
    policy = np.argmax(maxvalues)
    print("policy: ", policy+1)
    return (max(maxvalues), policy+1)

def updateCount(R, N, s, a, r, sn):
    N[s-1,a-1,sn-1] += 1
    R[s-1,a-1] += r
    
# Generate R and T functions
def RNupdate(S, A, R, N, data):
    for o in data:
        updateCount(R, N, o[0], o[1], o[2], o[3]) 
    for s in S:
        for a in A:
            n = sum(N[s-1,a-1,:])
            if n != 0:
                r = R[s-1,a-1]/n
                R[s-1,a-1] = r
            for sn in S:
                N[s-1,a-1,sn-1] = N[s-1,a-1,sn-1]/n
    return R, N


# Use Bellman updates to update state values and associated policies
def solve(model, k_max):
    U, P = model.U, model.P
    for k in range(k_max):
        val = [backup(model,s) for s in model.S]
        U =[e[0] for e in val] 
        model.U = U
    pol = [backup(model,s) for s in model.S]
    P = [e[1] for e in pol]
    model.P = P 
    print ("P =", P)


def main():
    #if len(sys.argv) != 3:
    #    raise Exception("usage: python project2_small.py <infile>.csv <outfile>.policy")

    # create a transition matrix
    

    
    
    
    Slen = 10    # |S|
    Alen = 10    # |A|
    eps = 0.99  # discount factor - given
    k_max = 200

    S = [i+1 for i in range(Slen)] # set of states
    A = [i+1 for i in range(Alen)] # set of actions
    T = np.zeros((Slen, Alen, Slen)) # transition function to state sn from state 's' when taking action 'a'
    R = np.zeros((Slen, Alen))       # Reward when taking action 'a' in state 's'
    O = np.zeros((Slen, Alen))  
    U = np.zeros(Slen)               # utility intialization
    P = [i+1 for i in range(Slen)]   # policy vector

    model = POMDP(S, A, T, R, O, eps, U, P) # initiate a maximum likelihood MDP
    
    solve (mlmdp, k_max)             # make k_max Bellman updates for each state.
 

    valuevector = mlmdp.U
    policyvector = mlmdp.P
    for row in policyvector:
        print(row)

    with open(outputfilename, "w") as file:
        for policy in policyvector:
            file.write(str(policy) + "\n")

    et = time.time()   # get the end time
    running_time = et - st  # get the execution time
    print("running time: ", running_time)

    # plot of the value/policy grid
    polmatrix = np.reshape(policyvector, (10,10))
    valmatrix = np.reshape(valuevector, (10,10))

    plt.pcolormesh(valmatrix)
    plt.title('matplotlib.pyplot.pcolormesh() function Example', fontweight ="bold")

    arrows = {2:(1,0), 1:(-1,0),3:(0,1), 4:(0,-1)}
    scale = 0.25


    for r, row in enumerate(polmatrix):
        for c, cell in enumerate(row):
            plt.arrow(c, 9-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.2)

    plt.show()

if __name__ == '__main__':
    main()