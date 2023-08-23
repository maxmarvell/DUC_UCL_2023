from path_integral import list_generator
from path_integral_thicker import get_tiles
from P_generator import Exponential_Map
import numpy as np 
import math

def main():
    W = np.loadtxt(f'data/FoldedTensors/DU_2_3806443768.csv',delimiter=',',dtype='complex_')
    P = np.loadtxt(f'data/FoldedPertubations/P_2_866021931.csv',delimiter=',',dtype='complex_')
    e = 1e-7
    q = 2
    d = 3

    G = Exponential_Map(e,P)
    PW = np.einsum('ab,bc->ac',G,W).reshape(q**2,q**2,q**2,q**2)

    tiles = get_tiles(PW,d)

    I, Z = np.zeros(q**2,dtype='complex_'), np.zeros(q**2,dtype='complex_')
    I[0], Z[1] = 1, 1

    a = np.einsum('a,b,c->abc',I,I,Z).reshape(-1)
    b = np.einsum('a,b,c->abc',Z,I,I).reshape(-1)

    print('\n',np.abs(path_integral(0.5,5.5,d,tiles,a,b)))

def path_integral(x:float,
                  t:float,
                  d:int,
                  tiles:dict,
                  a:np.ndarray,
                  b:np.ndarray):
    
    def transfer_matrix(a:np.ndarray,
                        l:int,
                        horizontal:bool = True,
                        terminate:bool = False):

        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b
        '''

        if horizontal:
            direct = tiles['h_direct']
            defect = tiles['h_defect']
        else:
            direct = tiles['v_direct']
            defect = tiles['v_defect']

        if not terminate:
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l-d):
                a = np.einsum('ba,b->a',direct,a)
            return np.einsum('ba,b->a',defect,a)

        elif terminate and (int(2*(t+x))%2 == 0):
            print('    TERMINATING WITH DIRECT')
            for _ in range(l):
                a = np.einsum('ba,b->a',direct,a)
            return np.einsum('a,a->',b,a)

        else:
            print('      TERMINATING WITH DEFECT')
            print(f'      Computing transfer matrix len {l}, Horizontal? {horizontal}')
            for _ in range(l-d):
                a = np.einsum('ba,b->a',direct,a)
            a = np.einsum('ba,b->a',defect,a)
            return np.einsum('a,a->',b,a)
        
    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram
        '''

        for i in range(len(v)):

            a = transfer_matrix(a,h[i])

            print(f'    Computed horizontal transfer matrix of length {h[i]}')

            a = transfer_matrix(a,v[i],horizontal=False)

            print(f'    Computed vertical transfer matrix of length {v[i]}')

        return transfer_matrix(a,h[-1],terminate=True) if len(h) > len(v) else np.einsum('a,a->',b,a)
    
    vertical_data = {}
    horizontal_data = {}

    x_h, x_v = math.ceil(t+x), math.floor(t+1-x)

    k = min(x_h//d,x_v//d)

    n = 1
    sum = 0

    while n < k:

        vertical_data = {}
        horizontal_data = {}

        x_h2 = x_h - n*(d-1)

        if int(2*(t+x))%2 == 0:
            x_v2 = x_v - (n)*(d-1) - d
        else:
            x_v2 = x_v - (n)*(d-1) - 1
        
        list_generator(x_v2,vertical_data,k=k)
        list_generator(x_h2,horizontal_data,k=k+1)

        print(f'\nComputing skeleton diagram for {n} turns!')

        try:
            l1, l2 = horizontal_data[n], vertical_data[n]
            for h in l1:
                for v in l2:
                    print(f'\n   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
        except:pass
            
        try:
            l1, l2 = horizontal_data[n+1], vertical_data[n]
            for h in l1:
                for v in l2:
                    print(f'\n   Diagram h:{h} and v:{v}')
                    sum += skeleton(h,v,a)
        except:pass
                        
        try:
            l1, v = horizontal_data[n], vertical_data[0]
            for h in l1:
                print(f'\n   Diagram h:{h} and v:{v}')
                sum += skeleton(h,[])
        except:pass

        n += 1

    return sum

if __name__ == '__main__':
    main()