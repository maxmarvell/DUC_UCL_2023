import numpy as np
from scipy.stats import unitary_group
from scipy.optimize import minimize
from itertools import permutations

'''
    TODO: check
'''

# class FoldedDUwithSoliton:

#     def __init__(self,q):
#         self.q = q
#         self.W = np.full(shape=[q**2,q**2,q**2,q**2],fill_value=np.nan)
#         self.W[0,0,0,0] = 1  

#         for i in  range(self.q**2):
#             for j in range(self.q**2):
#                 if i != 0 or j != 0: 
#                     self.W[i,j,0,0] = 0
#                     self.W[i,0,j,0] = 0
#                     self.W[0,0,i,j] = 0
#                     self.W[0,i,0,j] = 0

#     def update(self,x):


#     def constrain(self):
#         pass

def constrained_DU(q:int):

    def generate_W(q:int):

        W = np.full(shape=[q**2,q**2,q**2,q**2],fill_value=np.nan)
        W[0,0,0,0] = 1

        for i in  range(q**2):
            for j in range(q**2):
                if i != 0 or j != 0: 
                    W[i,j,0,0] = 0
                    W[i,0,j,0] = 0
                    W[0,0,i,j] = 0
                    W[0,i,0,j] = 0
        
        return W

    # for i in range(q**2):
    #     for j in range(q**2):
    #         print(f'{W[i,j,0]},{W[i,j,1]},{W[i,j,2]},{W[i,j,3]}')

    def soliton(x:np.ndarray, q:int):

        W = generate_W(q)

        for index, i in zip(np.argwhere(np.isnan(W)),x):
            W[*index] = i

        I, Z = np.zeros(q**2), np.zeros(q**2)
        I[0], Z[1] = 1, 1

        IZ = np.einsum('a,b->ab',I,Z)
        ZI = np.einsum('a,b->ab',Z,I)

        return np.linalg.norm(
            np.einsum('abcd,bd->ac',W,IZ)-1,
            np.einsum('abcd,bd->ac',W,ZI)-1)
    
    x0 = np.ones(q**8 - (4*q**4 - 4*q**2 + 1))

constrained_DU(2)
