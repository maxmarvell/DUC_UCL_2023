from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.linalg import expm 
import numpy as np
import pandas as pd
import re
import os

def list_generator(x:int,
                   data:dict,
                   k:int=np.inf,
                   lists:np.ndarray=[]):
    '''
        Generates a complete set of possible lists which can
        combine to form a complete set 
    '''

    if x == 0:
        try:
            data[len(lists)].append(lists)
        except:
            data[len(lists)] = [lists]
        return
    elif len(lists) >= k:
        return 

    for i in range(1,x+1):
        sublist = lists.copy()
        sublist.append(i)
        list_generator(x-i,data,k,sublist)

def randomise(P, R):
    return -abs(np.einsum("a,a->", P, R))

def Exponential_Map(e, P):
    return expm(e*P)

def fit(space, cross_section):

    def Gaussian(x,a,b,c):
        return a*np.exp(-((x-b)**2)/(2*c**2))

    def partial_gauss(a,b,c): 
        def P(x): return Gaussian(x,a,b,c)
        return P

    dense_inputs = np.linspace(space[0],space[-1],100)
    opt_pars, _ = curve_fit(Gaussian, xdata=space, ydata=cross_section)
    G = np.vectorize(partial_gauss(opt_pars[0],opt_pars[1],opt_pars[2]))
    fit_data = G(dense_inputs)

    R2 = r2_score(cross_section, G(space))

    return dense_inputs, fit_data, R2, opt_pars[2]

class Node:
    def __init__(self,data:complex=None):
        self.left = None
        self.right = None
        self.data = data

class Tree:
    def __init__(self,root=Node()):
        self.root = root
        self.current = self.root

    def move_left(self):
        if not self.current.left:
            self.current.left = Node()
        self.current = self.current.left

    def move_right(self):
        if not self.current.right:
            self.current.right = Node()
        self.current = self.current.right
    
    def assign_data(self,data:complex):
        self.current.data = data

    def return_root(self):
        self.current = self.root  

def random_circuit(tspan:int,
                   gates:np.ndarray,
                   temp:float):
    
    circuit = pd.DataFrame()

    for T in range(2*tspan):

        t = float(T)/2

        row = np.array([])
        inds = np.array([])

        for x in range(-T,T+1,2):
            x = float(x)/2
            inds = np.append(inds,x)
            inds = np.append(inds,x+0.5)
            gate = distribute(gates,temp)
            row = np.append(row,gate)
            row = np.append(row,gate)

        floquet = pd.Series(row,inds,name=t)

        circuit = pd.concat([circuit, floquet.to_frame().T])

    return circuit

def distribute(gates:np.ndarray,
               temp:float):

    def thermal(x:float):
        return np.exp(-x/temp)
    
    x = np.linspace(0,1,len(gates)+1)

    p = thermal(x[:-1]) - thermal(x[1:])

    p /= np.sum(p)

    return np.random.choice(gates,p=p)
        
def get_gates(q:int,
              e:float,
              pure:bool = False):

    PW, gates = dict(), list()

    rstr = f'DU_{q}' + r'_([0-9]*).csv'
    rx = re.compile(rstr)

    for _, _, files in os.walk(f'data/FoldedTensors'):
        for file in files:

            res = rx.match(file)

            if not res: continue

            seed = int(res.group(1))

            gates.append(seed)
    
            W = np.loadtxt(f'data/FoldedTensors/DU_{q}_{seed}.csv',
                           delimiter=',',dtype='complex_')
            
            if pure: 
                PW[seed] = W.reshape(q**2,q**2,q**2,q**2)
                continue

            P = np.loadtxt(f'data/FoldedPertubations/P_{q}_{seed}.csv',
                           delimiter=',',dtype='complex_')
            
            G = Exponential_Map(e,P)

            PW[seed] = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)
            
    return PW, gates