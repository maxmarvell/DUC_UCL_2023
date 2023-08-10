import numpy as np
from time import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm 



def main():

    q = 2
    e = 0.1
    W_path = "./Sample_Tensor.csv"
    P_path = "./Sample_Perturbation.csv"
    W = np.loadtxt(W_path,delimiter=',',dtype='complex_')
    P = np.loadtxt(P_path,delimiter=',',dtype='complex_')
    P = expm(1j*e*P)
    PW = np.einsum('ab,bc->ac',P,W).reshape([q**2,q**2,q**2,q**2])
    print("\n")

    T = 5
    canvas = path_integral(T,PW.reshape([q**2,q**2,q**2,q**2]))
    plt.imshow(np.abs(canvas),cmap='hot', interpolation='nearest')
    plt.show()
    #print(canvas)




def path_integral(T:float,W:np.ndarray):

    def transfer_matrix(X: float, W: np.ndarray, a: np.ndarray,
                        x: int, b: np.ndarray = [],
                        horizontal: bool = True,
                        terminate: bool = False):

        """
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        """

        if horizontal:
            direct = W[0,:,:,0]
            defect = W[:,0,:,0]
        else:
            direct = W[:,0,0,:]
            defect = W[0,:,0,:]

        if not terminate:
            for _ in range(x-1):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('ab,b->a',defect,a)

        elif terminate and (int(2*(T+X))%2 == 0):
            for _ in range(x):
                a = np.einsum('ab,b->a',direct,a)
            return np.einsum('a,a->',b,a)

        else:
            for _ in range(x-1):
                a = np.einsum('ab,b->a',direct,a)
            a = np.einsum('ab,b->a',defect,a)
            return np.einsum('a,a->',b,a)


        

    def skeleton(X:float, x_h:np.ndarray, x_v:np.ndarray, W:np.ndarray,
                a:np.ndarray, b:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        for i in range(len(x_v)):
            a = transfer_matrix(X,W,a,x_h[i])
            a = transfer_matrix(X,W,a,x_v[i],horizontal=False)

        return transfer_matrix(X,W,a,x_h[-1],b=b,terminate=True) if len(x_v) < len(x_h) else np.einsum('a,a->',a,b)


    def list_generator(x:int,data:dict,k:int=np.inf,
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



    canvas = np.full(shape=[int(2*T), int(4*T)-2], fill_value=0, dtype="complex_")
    canvas[0,int(2*T)-1] = 1.0
    a, b = np.array([0,1,0,0],dtype="complex_"), np.array([0,1,0,0],dtype="complex_")

    print(skeleton(-2.5,[1,1,2],[1,1,3],W,a,b))

    for t in range(1, int(2*T)):
        for x in range(-t, t):

            vertical_data = {}
            horizontal_data = {}
            x_h = math.ceil(t/2 - x/2)
            #print(x_h)
            x_v = math.floor(t/2 + 1 + x/2)
            #print(x_v)

            #k = min(x_h,x_v)
            #print(k)

            k = min(x_h, x_v)

            #k_v = min(x_h, x_v - 1)


            list_generator(x_h,horizontal_data,k=k)
            list_generator(x_v,vertical_data)

            #print(horizontal_data.keys())
            #print(vertical_data.keys())
            #print(k_h)

            n = 1
            sum = 0
            #print(k_h)

            while n <= k:

                try:
                    l1, l2 = horizontal_data[n], vertical_data[n]
                    for h in l1:
                        for v in l2:
                            sum += skeleton(-x/2,h,v,W,a,b)
                except:
                    print("exeception1")
                    pass
                    
                try:
                    l1, l2 = horizontal_data[n + 1], vertical_data[n]
                    for h in l1:
                        for v in l2:
                            sum += skeleton(-x/2,h,v,W,a,b)
                except:
                    print("exeception2")
                    pass
                                
                try:
                    l1, v = horizontal_data[n], vertical_data[0]
                    for h in l1:
                        sum += skeleton(-x/2,h,[],W,a,b)
                except:
                    print("exeception3")
                    pass

                n += 1
         
                print("\n")

            canvas[t,x+int(2*T)-1] = sum

    return canvas


if __name__ == '__main__':
    main()