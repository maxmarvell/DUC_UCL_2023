import numpy as np
from time import time
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm

def main():

    d = 3
    T = 24.0
    q = 2

    
    range = [5,5.5,6,6.5,7,
             7.5,8,8.5,9,9.5,
             10,10.5,11,11.5,12,
              12.5,13,13.5,14,14.5]


    for i in [0,1,2,3,4,5]:
        for e in range:
            Tile_dir = f"./Tile_Zoo_{i}/Tile_Zoo_{i}_e{e}"
            canvas = path_integral(T,d,q,Tile_dir)
            np.savetxt(f"./Larders/Larder_{i}/T{T}_d{d}_e{e}", canvas, delimiter=",")

            #plt.imshow(np.abs(canvas),cmap='hot', interpolation='nearest')
            #plt.show()

            #dimt, dimx = canvas.shape
            #charges = np.full(shape=[dimt],fill_value=0,dtype="complex_")
            #for t in range(dimt):
                #sum = 0.0
                #for x in range(dimx):
                    #sum += canvas[t,x]
                #charges[t] = sum
            #print(charges)
            #print("\n")


def path_integral(T:float, d:int, q:int, Tile_dir:str):

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



    canvas = np.full(shape=[int(2*T), int(4*T)-2], fill_value=0.0)
    canvas[0,int(2*T)-1] = 1.0

    Tile_Zoo = dict()
    Tile_Zoo["Main h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{d}x{d}",delimiter=',')
    Tile_Zoo["Main h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{d}x{d}",delimiter=',')
    Tile_Zoo["Main v_direct"] = np.loadtxt(f"{Tile_dir}/v_direct_{d}x{d}",delimiter=',')
    Tile_Zoo["Main v_defect"] = np.loadtxt(f"{Tile_dir}/v_defect_{d}x{d}",delimiter=',')


    for t in range(1, int(2*T)):
        for x in range(-t, t):

            vertical_data = {}
            horizontal_data = {}
            dim_h = math.ceil(t/2 - x/2)
            dim_v = math.floor(t/2 + 1 + x/2)
            k = min(dim_h, dim_v)

            if (dim_v%d) != 0:
                Tile_Zoo["Low Trim h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{d}x{dim_v%d}",delimiter=',')
                Tile_Zoo["Low Trim h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{d}x{dim_v%d}",delimiter=',')
                Tile_Zoo["Low Trim v_direct"] = np.loadtxt(f"{Tile_dir}/v_direct_{d}x{dim_v%d}",delimiter=',')
                Tile_Zoo["Low Trim v_defect"] = np.loadtxt(f"{Tile_dir}/v_defect_{d}x{dim_v%d}",delimiter=',')
                a = np.zeros(q**(2*(dim_v%d)))
                a[q**(2*(dim_v%d - 1))] = 1.0
                list_generator((dim_v//d + 1)-1,vertical_data)
            else:
                a = np.zeros(q**(2*d))
                a[q**(2*(d - 1))] = 1.0
                list_generator((dim_v//d)-1,vertical_data)
            

            if (dim_h%d) != 0:
                Tile_Zoo["Left Trim h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{dim_h%d}x{d}",delimiter=',')
                Tile_Zoo["Left Trim h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{dim_h%d}x{d}",delimiter=',')
                Tile_Zoo["Left Trim v_direct"] = np.loadtxt(f"{Tile_dir}/v_direct_{dim_h%d}x{d}",delimiter=',')
                Tile_Zoo["Left Trim v_defect"] = np.loadtxt(f"{Tile_dir}/v_defect_{dim_h%d}x{d}",delimiter=',')
                list_generator((dim_h//d + 1),horizontal_data,k=k)
            else:
                list_generator((dim_h//d),horizontal_data,k=k)


            if (dim_v%d) != 0 and (dim_h%d) != 0:
                Tile_Zoo["Corner h_direct"] = np.loadtxt(f"{Tile_dir}/h_direct_{dim_h%d}x{dim_v%d}",delimiter=',')
                Tile_Zoo["Corner h_defect"] = np.loadtxt(f"{Tile_dir}/h_defect_{dim_h%d}x{dim_v%d}",delimiter=',')



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

    return canvas


if __name__ == '__main__':
    main()


