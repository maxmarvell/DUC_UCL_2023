import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def main():

    range = [5,5.5,6,6.5,7,
             7.5,8,8.5,9,9.5,
             10,10.5,11,11.5,12,
             12.5,13,13.5,14,14.5]
    
    T1, T1err, T2, T2err, T3, T3err = [],[],[],[],[],[]
    T4, T4err, T5, T5err, T6, T6err = [],[],[],[],[],[]
    
    for e in range:

        path1 = f"./Larders/Larder_1/T24.0_d3_e{e}"
        path2 = f"./Larders/Larder_2/T24.0_d3_e{e}"
        path3 = f"./Larders/Larder_3/T24.0_d3_e{e}"
        path4 = f"./Larders/Larder_4/T24.0_d3_e{e}"
        path5 = f"./Larders/Larder_5/T24.0_d3_e{e}"
        path6 = f"./Larders/Larder_6/T24.0_d3_e{e}"

        T, Tstd = get_T(path1)
        T1.append(T)
        T1err.append(Tstd)

        T, Tstd = get_T(path2)
        T2.append(T)
        T2err.append(Tstd)

        T, Tstd = get_T(path3)
        T3.append(T)
        T3err.append(Tstd)

        T, Tstd = get_T(path4)
        T4.append(T)
        T4err.append(Tstd)

        T, Tstd = get_T(path5)
        T5.append(T)
        T5err.append(Tstd)

        T, Tstd = get_T(path6)
        T6.append(T)
        T6err.append(Tstd)


    avg = []
    std = []
    for i in np.arange(0,len(range)):
        column = [T1[i],T2[i],T3[i],T4[i],T5[i],T6[i]]
        std.append(np.nanstd(np.array(column)))
        avg.append(np.nanmean(np.array(column)))


    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel("$\epsilon$", size=14, fontname="Times New Roman", labelpad=10)
    plt.ylabel("T$_{sol}$" ,size=14, fontname="Times New Roman", labelpad=10)
    plt.title("Half Life of Soliton vs. Perturbation Strength", fontweight ='bold',size=14, fontname="Times New Roman")
    ax.minorticks_on()
    major_ticks = np.arange(range[0],range[-1],(range[-1]-range[0])*2/len(range))
    minor_ticks = np.arange(range[0],range[-1],(range[-1]-range[0])/(4*len(range)))
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor', linewidth=0.5, alpha=0.5)

    plt.plot(range,avg,marker="d",color='k',ms=5,linewidth=0.5,linestyle="dashed",label="Haar Average")
    plt.errorbar(range,T1,yerr=T1err,marker="d",color='k',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="1st seed")
    plt.errorbar(range,T2,yerr=T2err,marker="d",color='r',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="2nd seed")
    plt.errorbar(range,T3,yerr=T3err,marker="d",color='g',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="3rd seed")
    plt.errorbar(range,T4,yerr=T4err,marker="d",color='m',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="4th seed")
    plt.errorbar(range,T5,yerr=T5err,marker="d",color='c',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="5th seed")
    plt.errorbar(range,T6,yerr=T6err,marker="d",color='b',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="6th seed")

    plt.fill_between(range,np.array(avg)-np.array(std),np.array(avg)+np.array(std),color='k',alpha=0.1,linewidth=0.3)

    plt.grid()
    plt.legend()
    plt.show()



def exp_fit(time, weights):

    def Function(t,a,T):
        return a*(np.exp(-t/T))

    def partial_Function(a,T): 
        def P(t): return Function(t,a,T)
        return P

    dense_inputs = np.linspace(time[0],time[-1],100)
    opt_pars, cov_pars = curve_fit(Function, xdata=time, ydata=weights)
    astd = np.sqrt(float(np.diag(cov_pars)[0]))
    Tstd = np.sqrt(float(np.diag(cov_pars)[1]))
    G = np.vectorize(partial_Function(opt_pars[0],opt_pars[1]))
    fit_data = G(dense_inputs)  

    R2 = r2_score(weights, G(time))
    return opt_pars[1], Tstd


def get_T(path):

    canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
    dim_t, dim_x = np.shape(canvas)
    T = 24.0
    soliton = []
    time = np.arange(0.0,24.0,0.5)

    for t in range(int(2*T)):
        x = int(2*T)-t-1
        soliton.append((np.abs(canvas)**2)[t,x]) 

    T, Tstd = exp_fit(time, soliton)
    return T*np.log(2), Tstd*np.log(2)



if __name__ == '__main__':
    main()