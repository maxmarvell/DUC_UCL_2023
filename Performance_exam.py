import numpy as np
from time import time, sleep
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pandas as pd
import threading


def main():

    T = 24.0
    q = 2
    Tile_dir = "./Tile_Zoo_7_e10"
    runtime_store = pd.read_csv("./Runtime_Scores.csv").to_dict('list')
    charge_store = pd.read_csv("./Charge_Scores.csv").to_dict('list')

    '''''

    keys = ["1,3","1,6","1,12","1,k",
            "2,3","2,6","2,12","2,k",
            "3,3","3,6","3,12","3,k",
            "4,3","4,6","4,12","4,k"]

    d = [1,1,1,1,
        2,2,2,2,
        3,3,3,3,
        4,4,4,4]

    n_max = [3,6,12,np.nan,
            3,6,12,np.nan,
            3,6,12,np.nan,
            3,6,12,np.nan]

    '''''

    keys = ["4,k"]
    d = [4]
    n_max = [np.nan]

    for i in range(len(keys)):

        canvas, charges, runtimes = path_integral(T,d[i],q,Tile_dir,n_max[i])
        charge_store[keys[i]] = charges
        runtime_store[keys[i]] = runtimes
        print(runtime_store)
        print("\n")

    data1 = pd.DataFrame(runtime_store)
    data2 = pd.DataFrame(charge_store)
    data1.to_csv("./Runtime_Scores.csv")
    data2.to_csv("./Charge_Scores.csv")





def path_integral(T:float, d:int, q:int, Tile_dir:str, n_max:int=np.nan):

    def transfer_matrix(t:float, X:float, Tile_Zoo:dict, a: np.ndarray,
                        l:int, dim_h:int, dim_v:int, d:int,
                        horizontal:bool = True,
                        terminate:bool = False,
                        status:str = "Main"):

        """
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        """

        if horizontal:
            direct = Tile_Zoo[status + " h_direct"]
            defect = Tile_Zoo[status + " h_defect"]
        else:
            direct = Tile_Zoo[status + " v_direct"]
            defect = Tile_Zoo[status + " v_defect"]


        if not terminate:
            for i in range(l-1):

                if i == 0 and status == "Low Trim" and (dim_h%d) != 0:
                    a = np.einsum('ab,b->a',Tile_Zoo["Corner h_direct"],a)
                elif i != 0 and status == "Left Trim" and horizontal:
                    a = np.einsum('ab,b->a',Tile_Zoo["Main h_direct"],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)

            if l == 1 and status == "Low Trim" and (dim_h%d) != 0:
                return np.einsum('ab,b->a',Tile_Zoo["Corner h_defect"],a)  
            elif l > 1 and status == "Left Trim" and horizontal:
                return np.einsum('ab,b->a',Tile_Zoo["Main h_defect"],a)
            else:
                return np.einsum('ab,b->a',defect,a) 


        elif (int(2*(t+X))%2 == 0) and horizontal:
            for i in range(l):

                if i == 0 and status == "Low Trim" and (dim_h%d) != 0:
                    a = np.einsum('ab,b->a',Tile_Zoo[f"Corner h_direct"],a)
                elif i != 0 and status == "Left Trim":
                    a = np.einsum('ab,b->a',Tile_Zoo["Main h_direct"],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)
            return a[1]


        elif ((int(2*(t+X))%2 == 0) and not horizontal):
            for _ in range(l-1):

                a = np.einsum('ab,b->a',direct,a)

            a = np.einsum('ab,b->a',defect,a)
            return a[1]


        elif ((int(2*(t+X))%2 != 0) and not horizontal):
            for _ in range(l):

                a = np.einsum('ab,b->a',direct,a)
            return a[1]


        else:
            for i in range(l-1):

                if i == 0 and status == "Low Trim" and (dim_h%d) != 0:
                    a = np.einsum('ab,b->a',Tile_Zoo[f"Corner h_direct"],a)
                elif i != 0 and status == "Left Trim":
                    a = np.einsum('ab,b->a',Tile_Zoo["Main h_direct"],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)

            if l == 1 and status == "Low Trim" and (dim_h%d) != 0:
                a = np.einsum('ab,b->a',Tile_Zoo[f"Corner h_defect"],a)
                return a[1]

            elif l > 1 and status == "Left Trim":
                a = np.einsum('ab,b->a',Tile_Zoo["Main h_defect"],a)
                return a[1]

            else:
                a = np.einsum('ab,b->a',defect,a)
                return a[1]



    def skeleton(t:float, X:float, x_h:np.ndarray, x_v:np.ndarray, Tile_Zoo:dict,
                a:np.ndarray, dim_h:int, dim_v:int, d:int):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        if x_v == []:

            status = "Low Trim" if (dim_v%d) != 0 else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0) else "Main"        
            return transfer_matrix(t,X,Tile_Zoo,a,x_h[-1],dim_h,dim_v,d,terminate=True,status=status)
        
        elif len(x_v) == len(x_h):
            for i in range(len(x_v)-1):

                status = "Low Trim" if ((dim_v%d)!=0 and i==0) else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_h[i],dim_h,dim_v,d,status=status)

                status = "Left Trim" if ((dim_h%d)!=0 and x_h[0]==1 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_v[i],dim_h,dim_v,d,horizontal=False,status=status)

            status = "Low Trim" if ((dim_v%d)!=0 and len(x_v)==1) else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0 and len(x_v)==1) else "Main"
            a = transfer_matrix(t,X,Tile_Zoo,a,x_h[-1],dim_h,dim_v,d,status=status)

            status = "Left Trim" if ((dim_h%d)!=0 and x_h[0]==1 and len(x_v)==1) else "Main"
            return transfer_matrix(t,X,Tile_Zoo,a,x_v[-1],dim_h,dim_v,d,horizontal=False,terminate=True,status=status)

        else:
            for i in range(len(x_v)):

                status = "Low Trim" if ((dim_v%d)!=0 and i==0) else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_h[i],dim_h,dim_v,d,status=status)

                status = "Left Trim" if ((dim_h%d)!=0 and x_h[0]==1 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_v[i],dim_h,dim_v,d,horizontal=False,status=status)

            return transfer_matrix(t,X,Tile_Zoo,a,x_h[-1],dim_h,dim_v,d,terminate=True,status="Main")



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


    def countdown():
        global timer
        timer = 300
        for t in range(300):
            timer -= 1
            sleep(1)


    canvas = np.full(shape=[int(2*T), int(4*T)-2], fill_value=0, dtype="complex_")
    canvas[0,int(2*T)-1] = 1.0
    dimt, dimx = canvas.shape
    charges = np.full(shape=[dimt],fill_value=np.nan,dtype="complex_")
    charges[0] = 1.0
    runtimes = np.full(shape=[dimt],fill_value=np.nan)
    runtimes[0] = 0.0

    Tile_Zoo = dict()
    Tile_Zoo["Main h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{d}x{d}",delimiter=',',dtype='complex_')
    Tile_Zoo["Main h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{d}x{d}",delimiter=',',dtype='complex_')
    Tile_Zoo["Main v_direct"] = np.loadtxt(f"{Tile_dir}/v_direct_{d}x{d}",delimiter=',',dtype='complex_')
    Tile_Zoo["Main v_defect"] = np.loadtxt(f"{Tile_dir}/v_defect_{d}x{d}",delimiter=',',dtype='complex_')

    global start
    start = time()
    countdown_thread = threading.Thread(target=countdown)
    countdown_thread.start()

    while timer > 0:

        if timer == 0:
            break

        for t in range(1, int(2*T)):

            if timer == 0:
                break

            for x in range(-t, t):

                if timer == 0:
                    break

                vertical_data = {}
                horizontal_data = {}
                dim_h = math.ceil(t/2 - x/2)
                dim_v = math.floor(t/2 + 1 + x/2)
                
                if np.isnan(n_max):
                    k = min(dim_h, dim_v)
                else:
                    k = n_max

                if (dim_v%d) != 0:
                    Tile_Zoo["Low Trim h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{d}x{dim_v%d}",delimiter=',',dtype='complex_')
                    Tile_Zoo["Low Trim h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{d}x{dim_v%d}",delimiter=',',dtype='complex_')
                    Tile_Zoo["Low Trim v_direct"] = np.loadtxt(f"{Tile_dir}/v_direct_{d}x{dim_v%d}",delimiter=',',dtype='complex_')
                    Tile_Zoo["Low Trim v_defect"] = np.loadtxt(f"{Tile_dir}/v_defect_{d}x{dim_v%d}",delimiter=',',dtype='complex_')
                    a = np.zeros(q**(2*(dim_v%d)))
                    a[q**(2*(dim_v%d - 1))] = 1.0
                    list_generator((dim_v//d + 1)-1,vertical_data)
                else:
                    a = np.zeros(q**(2*d))
                    a[q**(2*(d - 1))] = 1.0
                    list_generator((dim_v//d)-1,vertical_data)
                

                if (dim_h%d) != 0:
                    Tile_Zoo["Left Trim h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{dim_h%d}x{d}",delimiter=',',dtype='complex_')
                    Tile_Zoo["Left Trim h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{dim_h%d}x{d}",delimiter=',',dtype='complex_')
                    Tile_Zoo["Left Trim v_direct"] = np.loadtxt(f"{Tile_dir}/v_direct_{dim_h%d}x{d}",delimiter=',',dtype='complex_')
                    Tile_Zoo["Left Trim v_defect"] = np.loadtxt(f"{Tile_dir}/v_defect_{dim_h%d}x{d}",delimiter=',',dtype='complex_')
                    list_generator((dim_h//d + 1),horizontal_data,k=k)
                else:
                    list_generator((dim_h//d),horizontal_data,k=k)


                if (dim_v%d) != 0 and (dim_h%d) != 0:
                    Tile_Zoo["Corner h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{dim_h%d}x{dim_v%d}",delimiter=',',dtype='complex_')
                    Tile_Zoo["Corner h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{dim_h%d}x{dim_v%d}",delimiter=',',dtype='complex_')


                n = 1
                sum = 0.0
        
                while n <= k:

                    try:
                        l1, l2 = horizontal_data[n], vertical_data[n]
                        for h in l1:
                            for v in l2:
                                sum += skeleton(t/2,-x/2,h,v,Tile_Zoo,a,dim_h,dim_v,d)
                    except Exception as e:
                        # Print the exception message
                        #print(f"An exception occurred: {str(e)}" + f" shape = {dim_h}x{dim_v}")
                        pass
                        
                    try:
                        l1, l2 = horizontal_data[n + 1], vertical_data[n]
                        for h in l1:
                            for v in l2:
                                sum += skeleton(t/2,-x/2,h,v,Tile_Zoo,a,dim_h,dim_v,d)
                    except Exception as e:
                        # Print the exception message
                        #print(f"An exception occurred: {str(e)}" + f" shape = {dim_h}x{dim_v}")
                        pass

                    if n == 1:               
                        try:
                            l1, v = horizontal_data[n], vertical_data[0]
                            for h in l1:
                                sum += skeleton(t/2,-x/2,h,[],Tile_Zoo,a,dim_h,dim_v,d)
                        except Exception as e:
                            # Print the exception message
                            #print(f"An exception occurred: {str(e)}" + f" shape = {dim_h}x{dim_v}")
                            pass

                    n += 1
            

                canvas[t,x+int(2*T)-1] = sum

            c = 0.0
            for i in range(dimx):
                c += canvas[t,i]
            charges[t] = c
            runtimes[t] = time() - start

        break


    return canvas, charges, runtimes


if __name__ == '__main__':
    main()


