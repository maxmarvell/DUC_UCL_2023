from path_integral import path_integral, list_generator
import numpy as np
import math


'''
    File to determine time point where there is no
    longer any solitonic behaviour then take the 
    time slice of the entire light cone at this point
'''

def main():

    seed = 3806443768
    q = 2
    e = 5e-7

    W = np.loadtxt(f'./data/FoldedTensors/DU_{q}_{seed}.csv',
                            delimiter=',',dtype='complex_')
    
    P = np.loadtxt(f'./data/FoldedPertubations/P_{q}_{seed}_{e}.csv',
                            delimiter=',',dtype='complex_')

    PW = np.einsum('ab,bc->ac',P,W).reshape(q**2,q**2,q**2,q**2)

    correlation = 1
    t = 0

    while correlation > 0.6:
        new = path_integral(t,t,PW)
        if new < correlation: correlation = new
        t += 1

    inds,data = np.array([]),np.array([])
    
    for x in range(-2*int(t),2*int(t)+1):
        x = float(x)/2
        inds = np.append(inds,x)
        data = np.append(data,[path_integral(x,t,PW)])

    print(data,t)

if __name__ == '__main__':
    main()