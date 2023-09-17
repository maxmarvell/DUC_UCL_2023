import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def main():

    range = [5,5.5,6,6.5,7,
             7.5,8,8.5,9,9.5,
             10,10.5,11,11.5,12,
             12.5,13,13.5,14,14.5]

    p1, p1err, p2, p2err, p3, p3err = [],[],[],[],[],[]
    p4, p4err, p5, p5err, p6, p6err = [],[],[],[],[],[]

    for e in range:

        path1 = f"./Larders/Larder_1/T24.0_d3_e{e}"
        path2 = f"./Larders/Larder_2/T24.0_d3_e{e}"
        path3 = f"./Larders/Larder_3/T24.0_d3_e{e}"
        path4 = f"./Larders/Larder_4/T24.0_d3_e{e}"
        path5 = f"./Larders/Larder_5/T24.0_d3_e{e}"
        path6 = f"./Larders/Larder_6/T24.0_d3_e{e}"

        P, Pstd = get_p(path1)
        p1.append(P)
        p1err.append(Pstd)

        P, Pstd = get_p(path2)
        p2.append(P)
        p2err.append(Pstd)

        P, Pstd = get_p(path3)
        p3.append(P)
        p3err.append(Pstd)

        P, Pstd = get_p(path4)
        p4.append(P)
        p4err.append(Pstd)

        P, Pstd = get_p(path5)
        p5.append(P)
        p5err.append(Pstd)

        P, Pstd = get_p(path6)
        p6.append(P)
        p6err.append(Pstd)

    avg = []
    std = []
    for i in np.arange(0,len(range)):
        column = [p1[i],p2[i],p3[i],p4[i],p5[i],p6[i]]
        std.append(np.nanstd(np.array(column)))
        avg.append(np.nanmean(np.array(column)))


    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel("$\epsilon$", size=14, fontname="Times New Roman", labelpad=10)
    plt.ylabel("p" ,size=14, fontname="Times New Roman", labelpad=10)
    plt.title("Exponent from Power Law Fit of $\sigma$(t) vs. Perturbation Strength", fontweight ='bold',size=14, fontname="Times New Roman")
    ax.minorticks_on()
    major_ticks = np.arange(range[0],range[-1],(range[-1]-range[0])/len(range))
    minor_ticks = np.arange(range[0],range[-1],(range[-1]-range[0])/(4*len(range)))
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor', linewidth=0.5, alpha=0.5)

    plt.errorbar(range,p1,yerr=p1err,marker="d",color='k',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="1st seed")
    plt.errorbar(range,p2,yerr=p2err,marker="d",color='r',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="2nd seed")
    plt.errorbar(range,p3,yerr=p3err,marker="d",color='g',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="3rd seed")
    plt.errorbar(range,p4,yerr=p4err,marker="d",color='m',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="4th seed")
    plt.errorbar(range,p5,yerr=p5err,marker="d",color='c',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="5th seed")
    plt.errorbar(range,p6,yerr=p6err,marker="d",color='b',ms=2,linewidth=0.2,linestyle="dashed",elinewidth=0.3,capsize=1.0,label="6th seed")
    
    plt.plot(range,avg,marker="v",color='k',ms=5,linewidth=0.3,linestyle="dashed",label="Haar Average")
    plt.fill_between(range,np.array(avg)-np.array(std),np.array(avg)+np.array(std),color='k',alpha=0.1,linewidth=0.3)

    plt.grid()
    plt.legend()
    plt.show()





def fit(space, cross_section):

    def Gaussian(x,a,b,c):
        return a*np.exp(-((x-b)**2)/(2*c**2))

    def partial_gauss(a,b,c): 
        def P(x): return Gaussian(x,a,b,c)
        return P

    dense_inputs = np.linspace(space[0],space[-1],100)
    opt_pars, cov_pars = curve_fit(Gaussian, xdata=space, ydata=cross_section)
    G = np.vectorize(partial_gauss(opt_pars[0],opt_pars[1],opt_pars[2]))
    fit_data = G(dense_inputs)

    R2 = r2_score(cross_section, G(space))

    return R2, opt_pars[2]


def power_fit(time, widths):

    def Function(x,a,p,b):
        return a*((x-time[0])**p) + b

    def partial_Function(a,p,b): 
        def P(x): return Function(x,a,p,b)
        return P

    dense_inputs = np.linspace(time[0],time[-1],100)
    opt_pars, cov_pars = curve_fit(Function, xdata=time, ydata=widths)
    astd = np.sqrt(float(np.diag(cov_pars)[0]))
    pstd = np.sqrt(float(np.diag(cov_pars)[1]))
    bstd = np.sqrt(float(np.diag(cov_pars)[2]))
    G = np.vectorize(partial_Function(opt_pars[0],opt_pars[1],opt_pars[2]))
    Gu = np.vectorize(partial_Function(opt_pars[0]+astd,opt_pars[1]+pstd,opt_pars[2]+bstd))
    Gl = np.vectorize(partial_Function(opt_pars[0]-astd,opt_pars[1]-pstd,opt_pars[2]-bstd))
    fit_data = G(dense_inputs)
    extreme_u = Gu(dense_inputs)
    extreme_l = Gl(dense_inputs)    

    R2 = r2_score(widths, G(time))

    return opt_pars[1], pstd


def get_p(path):

    canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
    dim_t, dim_x = np.shape(canvas)
    time = []
    widths = []
    t = -1
    R2 = 1.0

    while R2 >= 0.99:
        try:
            cross_section = list(np.abs(canvas[t,:])**2)
            space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)
            cross_section = cross_section[::2]
            space = space[::2]
            R2, width = fit(space, cross_section)
            if not R2 >= 0.99:
                break 
            time.append((dim_t+t)/2)
            widths.append(width)
            t -= 1
        except:
            break

    time = list(np.flip(np.array(time)))
    widths = list(np.flip(np.array(widths)))
    if len(widths) >= 15:
        p, pstd = power_fit(time, widths)
        return p, pstd
    else:
        return np.nan, np.nan


if __name__ == '__main__':
    main()