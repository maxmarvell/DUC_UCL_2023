from du_quantum_kit.tensors import *
from du_quantum_kit.circuit import *
import numpy as np
import pandas as pd

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


class RandomCircuit():

    def __init__(self,
                 q:int,
                 e:float,
                 tspan:int,
                 temp:float):
        
        self.circuit = pd.DataFrame()
        self.q = q
        self.e = e
        self.tspan = tspan
        self.temp = temp



    def generate_gates(self,
                       num:int,
                       save:bool = False,
                       dir:str = None):
        
        self.gates = []

        for _ in range(num):
            gate = FoldedGate(self.q)
            if save: gate.save(dir)
            gate.W = gate.W.reshape(self.q**4,self.q**4)
            self.gates.append(np.einsum('ab,bc->ac',exp_map(self.e,gate.P),gate.W).reshape(self.q**2,self.q**2,self.q**2,self.q**2))


        
    def return_gates(self,
                     dir:str):
        
        self.gates = []
        
        for (_,_,unitaries), (_,_,pertubations) in zip(os.walk(f'{dir}/dualunitaries'),os.walk(f'{dir}/pertubations')):
            for fileW, fileP in zip(unitaries,pertubations):
                W = np.loadtxt(f'{dir}/dualunitaries/{fileW}',delimiter=',',dtype='complex_')
                P = np.loadtxt(f'{dir}/pertubations/{fileP}',delimiter=',',dtype='complex_')
                self.gates.append(np.einsum('ab,bc->ac',exp_map(self.e,P),W).reshape(self.q**2,self.q**2,self.q**2,self.q**2))



    def random_circuit(self):

        for T in range(2*self.tspan):

            t = float(T)/2

            row = np.array([])
            inds = np.array([])

            for x in range(-T,T+1,2):
                x = float(x)/2
                gate = self.distribute()
                inds = np.append(inds,[x,x+0.5])
                row = np.append(row,[gate,gate])

            floquet = pd.Series(row,inds,name=t)

            self.circuit = pd.concat([self.circuit, floquet.to_frame().T])
    
    def distribute(self):

        def thermal(x:float):
            return np.exp(-x/self.temp)
        
        x = np.linspace(0,1,len(self.gates)+1)

        p = thermal(x[:-1]) - thermal(x[1:])

        p /= np.sum(p)

        return int(np.random.choice(range(len(self.gates)),p=p))
    

    def infinite_temperature(self):
        
        def path_integral(x:float,
                          t:float):
    
            def transfer_matrix(a:np.ndarray,
                                l:int,
                                X:float,
                                T:float,
                                horizontal:bool = True):

                if ((int(2*(t+x))%2 == 0) and horizontal) or ((int(2*(t+x))%2 != 0) and not horizontal):  
                    for _ in range(l):
                        
                        W = self.gates[self.circuit.loc[T,X]]

                        if horizontal:
                            X = X + 0.5
                            direct = W[0,:,:,0]
                            tree.move_right()
                        else:
                            X = X - 0.5
                            direct = W[:,0,0,:]
                            tree.move_left()

                        T = T + 0.5

                        a = np.einsum('ab,b->a',direct,a)

                    return a[1]

                for _ in range(l-1):

                    W = self.gates[self.circuit.loc[T,X]]

                    if horizontal:
                        X = X + 0.5
                        direct = W[0,:,:,0]
                        tree.move_right()
                    else:
                        X = X - 0.5
                        direct = W[:,0,0,:]
                        tree.move_left()

                    T = T + 0.5
                        
                    a = np.einsum('ab,b->a',direct,a)

                W = self.gates[self.circuit.loc[T,X]]

                if horizontal:
                    tree.move_right()
                    defect = W[:,0,:,0]
                else:
                    tree.move_left()
                    defect = W[0,:,0,:]

                a = np.einsum('ab,b->a',defect,a)
                tree.assign_data(a)

                return a[1]

            def search_tree(l:int,
                            horizontal:bool = True):
                
                if horizontal:
                    for _ in range(l):
                        tree.move_right()
                else:
                    for _ in range(l):
                        tree.move_left()

            def skeleton(h:np.ndarray,
                        v:np.ndarray,
                        a:np.ndarray):

                tree.return_root()
                X,T = 0,0

                if v == []:
                    return transfer_matrix(a,h[-1],X,T)

                elif len(v) == len(h):

                    for i in range(len(v)-1):

                        search_tree(h[i])
                        X += (h[i] - 1) / 2
                        T += h[i]/2

                        search_tree(v[i],horizontal=False)
                        X -= (h[i] - 1) / 2
                        T += v[i]/2

                    search_tree(h[-1])
                    X += (h[-1] - 1) / 2
                    T += h[-1]/2

                    a = tree.current.data

                    return transfer_matrix(a,v[-1],X,T,horizontal=False)

                elif len(h) > len(v):

                    for i in range(len(v)):

                        search_tree(h[i])
                        X += (h[i] - 1) / 2
                        T += h[i]/2

                        search_tree(v[i],horizontal=False)
                        X -= (v[i] - 1) / 2
                        T += v[i]/2

                    a = tree.current.data

                    return transfer_matrix(a,h[-1],X,T)

            a = np.zeros(self.q**2,dtype='complex_')
            a[1] = 1

            if x == 0 and t == 0.:
                a[1]

            x_h, x_v = math.ceil(t+x), math.floor(t+1-x)
            
            k = min(x_h,x_v)

            vertical_data = path_set(x_v-1,k=k)
            horizontal_data = path_set(x_h,k=k+1)

            n = 0
            sum = 0

            while n <= k:

                try:
                    l1, l2 = horizontal_data[n], vertical_data[n]
                    for h in l1:
                        for v in l2:
                            sum += skeleton(h,v,a)
                except:pass
                    
                try:
                    l1, l2 = horizontal_data[n+1], vertical_data[n]
                    for h in l1:
                        for v in l2:
                            sum += skeleton(h,v,a)
                except:pass

                n += 1

            return sum

        tree = Tree()
        df = pd.DataFrame()
        err = pd.Series()

        for T in range(2*self.tspan+1):
            t = float(T)/2

            data = np.array([])
            inds = np.array([])

            for x in range(-T+1,T+1):
                x = float(x)/2
                inds = np.append(inds,x)
                data = np.append(data,[path_integral(x,t)])

            print(f'computed for T = {t}s')

            s = pd.Series(np.abs(data),inds,name=t)        
            df = pd.concat([df, s.to_frame().T])
            err[t] = np.abs(sum(data))

        df = df.reindex(sorted(df.columns,key=lambda num: float(num)), axis=1)
        df = df.fillna(0)
        df = df.iloc[::-1]
        print(df,'\n')

