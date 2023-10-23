from utils import *
import numpy as np
import math

def PATH(x:float,
         t:float,
         q:int,
         d:int,
         TileZoo:dict,
         seed:int,
         e:float):

    def transfer_matrix(a:np.ndarray,
                        l:int,
                        horizontal:bool = True,
                        terminate:bool = False,
                        status:str = 'Main'):

        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        '''

        if horizontal:
            direct = TileZoo[status + ' h_direct']
            defect = TileZoo[status + ' h_defect']
        else:
            direct = TileZoo[status + ' v_direct']
            defect = TileZoo[status + ' v_defect']

        if not terminate:
            for i in range(l-1):

                if i == 0 and status == 'Low Trim' and (x_h%d) != 0:
                    a = np.einsum('ab,b->a',TileZoo['Corner h_direct'],a)
                elif i != 0 and status == 'Left Trim' and horizontal:
                    a = np.einsum('ab,b->a',TileZoo['Main h_direct'],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)

            if l == 1 and status == 'Low Trim' and (x_h%d) != 0:
                return np.einsum('ab,b->a',TileZoo['Corner h_defect'],a)  
            elif l > 1 and status == 'Left Trim' and horizontal:
                return np.einsum('ab,b->a',TileZoo['Main h_defect'],a)
            
            return np.einsum('ab,b->a',defect,a) 

        elif (int(2*(t+x))%2 == 0) and horizontal:
            for i in range(l):
                if i == 0 and status == 'Low Trim' and (x_h%d) != 0:
                    a = np.einsum('ab,b->a',TileZoo[f'Corner h_direct'],a)
                elif i != 0 and status == 'Left Trim':
                    a = np.einsum('ab,b->a',TileZoo['Main h_direct'],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)
            return a[1]

        elif ((int(2*(t+x))%2 == 0) and not horizontal):
            for _ in range(l-1):
                a = np.einsum('ab,b->a',direct,a)
            a = np.einsum('ab,b->a',defect,a)
            return a[1]

        elif ((int(2*(t+x))%2 != 0) and not horizontal):
            for _ in range(l):
                a = np.einsum('ab,b->a',direct,a)
            return a[1]

        for i in range(l-1):

            if i == 0 and status == 'Low Trim' and (x_h%d) != 0:
                a = np.einsum('ab,b->a',TileZoo[f'Corner h_direct'],a)
            elif i != 0 and status == 'Left Trim':
                a = np.einsum('ab,b->a',TileZoo['Main h_direct'],a)
            else:
                a = np.einsum('ab,b->a',direct,a)

        if l == 1 and status == 'Low Trim' and (x_h%d) != 0:
            a = np.einsum('ab,b->a',TileZoo[f'Corner h_defect'],a)
            return a[1]

        elif l > 1 and status == 'Left Trim':
            a = np.einsum('ab,b->a',TileZoo['Main h_defect'],a)
            return a[1]
        
        a = np.einsum('ab,b->a',defect,a)
        return a[1]

    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        if v == []:
            status = 'Low Trim' if (x_v%d) != 0 else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0) else 'Main'        
            return transfer_matrix(a,h[-1],terminate=True,status=status)
        
        elif len(v) == len(h):
            for i in range(len(v)-1):

                status = 'Low Trim' if ((x_v%d)!=0 and i==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and i==0) else 'Main'
                a = transfer_matrix(a,h[i],status=status)

                status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and i==0) else 'Main'
                a = transfer_matrix(a,v[i],horizontal=False,status=status)

            status = 'Low Trim' if ((x_v%d)!=0 and len(v)==1) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and len(v)==1) else 'Main'
            a = transfer_matrix(a,h[-1],status=status)

            status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and len(v)==1) else 'Main'
            return transfer_matrix(a,v[-1],horizontal=False,terminate=True,status=status)

        for i in range(len(v)):

            status = 'Low Trim' if ((x_v%d)!=0 and i==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and i==0) else 'Main'
            a = transfer_matrix(a,h[i],status=status)

            status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and i==0) else 'Main'
            a = transfer_matrix(a,v[i],horizontal=False,status=status)

        return transfer_matrix(a,h[-1],terminate=True,status='Main')

    vertical_data, horizontal_data = {}, {}
    x_h,x_v = math.ceil(t+x), math.floor(t+1-x)

    k = min(x_h,x_v)

    if (x_v%d) != 0:

        TileZoo['Low Trim h_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_direct_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Low Trim h_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_defect_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Low Trim v_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_direct_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Low Trim v_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_defect_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')

        a = np.zeros(q**(2*(x_v%d)))
        a[q**(2*(x_v%d - 1))] = 1.0
        list_generator((x_v//d + 1)-1,vertical_data)
    else:
        a = np.zeros(q**(2*d))
        a[q**(2*(d - 1))] = 1.0
        list_generator((x_v//d)-1,vertical_data)
    
    if (x_h%d) != 0:

        TileZoo['Left Trim h_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_direct_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Left Trim h_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_defect_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Left Trim v_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_direct_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Left Trim v_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_defect_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')

        list_generator((x_h//d + 1),horizontal_data,k=k)
    else:
        list_generator((x_h//d),horizontal_data,k=k)

    if (x_v%d) != 0 and (x_h%d) != 0:

        TileZoo['Corner h_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_direct_{x_h%d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Corner h_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_defect_{x_h%d}x{x_v%d}.csv',delimiter=',',dtype='complex_')

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

def tree_path(x:float,
              t:float,
              q:int,
              d:int,
              TileZoo:dict,
              seed:int,
              e:float,
              tree:Tree):
    
    def transfer_matrix(a:np.ndarray,
                        l:int,
                        horizontal:bool = True,
                        status:str = 'Main'):

        '''
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        '''

        if horizontal:
            direct = TileZoo[status + ' h_direct']
            defect = TileZoo[status + ' h_defect']
        else:
            direct = TileZoo[status + ' v_direct']
            defect = TileZoo[status + ' v_defect']

        if (int(2*(t+x))%2 == 0) and horizontal:
            for i in range(l):
                if i == 0 and status == 'Low Trim' and (x_h%d) != 0:
                    a = np.einsum('ab,b->a',TileZoo[f'Corner h_direct'],a)
                elif i != 0 and status == 'Left Trim':
                    a = np.einsum('ab,b->a',TileZoo['Main h_direct'],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)
            return a[1]

        elif ((int(2*(t+x))%2 == 0) and not horizontal):
            for _ in range(l-1):
                a = np.einsum('ab,b->a',direct,a)
            a = np.einsum('ab,b->a',defect,a)
            search_tree(l,status,horizontal)
            tree.assign_data(a)
            return a[1]

        elif ((int(2*(t+x))%2 != 0) and not horizontal):
            for _ in range(l):
                a = np.einsum('ab,b->a',direct,a)
            return a[1]


        for i in range(l-1):
            if i == 0 and status == 'Low Trim' and (x_h%d) != 0:
                a = np.einsum('ab,b->a',TileZoo[f'Corner h_direct'],a)
            elif i != 0 and status == 'Left Trim':
                a = np.einsum('ab,b->a',TileZoo['Main h_direct'],a)
            else:
                a = np.einsum('ab,b->a',direct,a)

        if l == 1 and status == 'Low Trim' and (x_h%d) != 0:
            a = np.einsum('ab,b->a',TileZoo[f'Corner h_defect'],a)
            search_tree(l,status)
            tree.assign_data(a)
            return a[1]

        elif l > 1 and status == 'Left Trim':
            a = np.einsum('ab,b->a',TileZoo['Main h_defect'],a)
            search_tree(l,status)
            tree.assign_data(a)
            return a[1]
            
        a = np.einsum('ab,b->a',defect,a)
        search_tree(l,status)
        tree.assign_data(a)
        return a[1]
    
    def search_tree(l:int,
                    status:str,
                    horizontal:bool = True):   
        
        if status == 'Low Trim':
            for _ in range(l*d+x_h%d-d*int(bool(x_h%d))):
                tree.move_right()
            for _ in range(x_v%d-1):
                tree.move_left()
            return
        elif status == 'Left Trim' and horizontal:
            for _ in range(l*d+x_h%d-d*int(bool(x_h%d))):
                tree.move_right()
            for _ in range(d-1):
                tree.move_left()
            return
        elif status == 'Left Trim':
            for _ in range(l*d):
                tree.move_left()
            for _ in range(x_h%d-1):
                tree.move_right()
            return
        elif status == 'Main' and horizontal:
            for _ in range(l*d):
                tree.move_right()
            for _ in range(d-1):
                tree.move_left()
            return
        
        for _ in range(l*d):
            tree.move_left()
        for _ in range(d):
            tree.move_right()
        

    def skeleton(h:np.ndarray,
                 v:np.ndarray,
                 a:np.ndarray):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        tree.return_root()
        
        if len(v) == len(h):
            for i in range(len(v)-1):
                status = 'Low Trim' if ((x_v%d)!=0 and i==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and i==0) else 'Main'
                search_tree(h[i],status)

                status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and i==0) else 'Main'
                search_tree(v[i],status,horizontal=False)

            status = 'Low Trim' if ((x_v%d)!=0 and len(v)==1) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and len(v)==1) else 'Main'
            search_tree(h[-1],status)
            a = tree.current.data

            status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and len(v)==1) else 'Main'
            return transfer_matrix(a,v[-1],horizontal=False,status=status)

        for i in range(len(v)):
            status = 'Low Trim' if ((x_v%d)!=0 and i==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and i==0) else 'Main'
            search_tree(h[i],status)

            status = 'Left Trim' if ((x_h%d)!=0 and h[0]==1 and i==0) else 'Main'
            search_tree(v[i],status,horizontal=False)

        status = 'Low Trim' if ((x_v%d) != 0 and len(v)==0) else 'Left Trim' if ((x_h%d)!=0 and (x_v%d)==0 and len(v)==0) else 'Main'
        if len(h) == 1:
            return transfer_matrix(a,h[-1],status=status)
        a = tree.current.data
        return transfer_matrix(a,h[-1],status=status)

    if x == 0 and t == 0.:
        return a[1]

    vertical_data, horizontal_data = {}, {}
    x_h,x_v = math.ceil(t+x), math.floor(t+1-x)

    k = min(x_h,x_v)

    if (x_v%d) != 0:

        TileZoo['Low Trim h_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_direct_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Low Trim h_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_defect_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Low Trim v_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_direct_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Low Trim v_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_defect_{d}x{x_v%d}.csv',delimiter=',',dtype='complex_')

        a = np.zeros(q**(2*(x_v%d)))
        a[q**(2*(x_v%d - 1))] = 1.0
        list_generator((x_v//d + 1)-1,vertical_data)
    else:
        a = np.zeros(q**(2*d))
        a[q**(2*(d - 1))] = 1.0
        list_generator((x_v//d)-1,vertical_data)
    
    if (x_h%d) != 0:

        TileZoo['Left Trim h_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_direct_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Left Trim h_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_defect_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Left Trim v_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_direct_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Left Trim v_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/v_defect_{x_h%d}x{d}.csv',delimiter=',',dtype='complex_')

        list_generator((x_h//d + 1),horizontal_data,k=k)
    else:
        list_generator((x_h//d),horizontal_data,k=k)

    if (x_v%d) != 0 and (x_h%d) != 0:

        TileZoo['Corner h_direct'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_direct_{x_h%d}x{x_v%d}.csv',delimiter=',',dtype='complex_')
        TileZoo['Corner h_defect'] = np.loadtxt(f'data/TileZoo/{q}_{seed}_{e}/h_defect_{x_h%d}x{x_v%d}.csv',delimiter=',',dtype='complex_')

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