import numpy as np
#from randomDUwithSoliton import finddUconstrained
import matplotlib.pyplot as plt

def main():

    q = 2
    Z = [1,0,0,-1]
    t = 100

    U = [1,0,0,0,0,0,-1j,0,0,-1j,0,0,0,0,0,1]
    compute_LC_correlation(q,U,Z,t)

def compute_LC_correlation(q: int, U, Z, t: int, type: str = 'positive'):

    '''
        Computes the correlation function of a local oper-
        ator along the light cone of DU circuit.Then plots the 
        correlation function as a function of time
    '''

    correlation = []
    local = channel_map(q,U,Z)
    correlation.append(innerproduct(q,local,Z))

    for _ in range(1,t):
        local = channel_map(q,U,local)
        sol = innerproduct(q,local,Z)
        correlation.append(np.abs(sol))

    plt.plot(range(t),correlation)
    plt.show()

def channel_map(q: int, U, Z, type: str = 'positive'):

    '''
        For each time step the M operator is applied to evolve the local
        operator in the Heisenberg picture.

        The M operator either acts in the positive or negative direction
        dependent on the type variable, default positive
    '''

    U = np.reshape(U,[q,q,q,q])
    Z = np.reshape(Z,[q,q])
    I = np.identity(q)

    if type == 'positive':
        state = np.einsum('ac,bd->abcd',Z,I)
        evolve = np.einsum('abcd,cdef,efgh->abgh',np.conj(U),state,U)
        Z = 1/q*np.einsum('abad->bd',evolve)    
    else:
        state = np.einsum('ac,bd->abcd',I,Z)
        evolve = np.einsum('abcd,cdef,efgh->abgh',np.conj(U),state,U)
        Z = 1/q*np.einsum('abcb->ac',evolve) 
    
    return(Z)

def innerproduct(q: int,b,a):

    '''
        A function that computes the inner product between two local
        operators a and b
    '''

    a = np.reshape(a,[q,q])
    pretracial = np.einsum('ab,bc->ac',b,a)
    return 1/q*np.einsum('ii->',pretracial)

if __name__ == '__main__':
    #U = finddUconstrained(2,soliton=True)
    main()
