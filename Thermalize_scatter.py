import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def main():

    range = [5,5.5,6,6.5,7,
             7.5,8,8.5,9,9.5,
             10,10.5,11,11.5,12,
             12.5,13,13.5,14,14.5]

    t1, t2, t3, t4, t5, t6 = [],[],[],[],[],[]

    for e in range:

        path1 = f"./Larders/Larder_1/T24.0_d3_e{e}"
        path2 = f"./Larders/Larder_2/T24.0_d3_e{e}"
        path3 = f"./Larders/Larder_3/T24.0_d3_e{e}"
        path4 = f"./Larders/Larder_4/T24.0_d3_e{e}"
        path5 = f"./Larders/Larder_5/T24.0_d3_e{e}"
        path6 = f"./Larders/Larder_6/T24.0_d3_e{e}"

        T = get_p(path1)
        t1.append(T)

        T = get_p(path2)
        t2.append(T)

        T = get_p(path3)
        t3.append(T)

        T = get_p(path4)
        t4.append(T)

        T = get_p(path5)
        t5.append(T)

        T = get_p(path6)
        t6.append(T)

    avg = list((np.array(t1)+np.array(t2)+np.array(t3)+np.array(t4)+np.array(t5)+np.array(t6))/6)
    std = []
    for i in np.arange(0,len(t1)):
        column = [t1[i],t2[i],t3[i],t4[i],t5[i],t6[i]]
        std.append(np.std(np.array(column)))


    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel("$\epsilon$", size=14, fontname="Times New Roman", labelpad=10)
    plt.ylabel("T$_{therm}$" ,size=14, fontname="Times New Roman", labelpad=10)
    plt.title("Thermalization Time vs. Perturbation Strength", fontweight ='bold',size=14, fontname="Times New Roman")
    ax.minorticks_on()
    major_ticks = np.arange(range[0],range[-1],(range[-1]-range[0])*2/len(range))
    minor_ticks = np.arange(range[0],range[-1],(range[-1]-range[0])/(4*len(range)))
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor', linewidth=0.5, alpha=0.5)

    plt.plot(range,avg,marker="d",color='k',ms=5,linewidth=0.5,linestyle="dashed",label="Haar Average")
    #plt.scatter(range,t1,marker="v",color='k',s=3,label="1st Seed")
    #plt.scatter(range,t2,marker="v",color='r',s=3,label="2nd Seed")
    #plt.scatter(range,t3,marker="v",color='g',s=3,label="3rd Seed")
    #plt.scatter(range,t4,marker="v",color='b',s=3,label="4th Seed")
    #plt.scatter(range,t5,marker="v",color='m',s=3,label="5th Seed")
    #plt.scatter(range,t6,marker="v",color='c',s=3,label="6th Seed")

    plt.plot(range,t1,marker="d",color='k',ms=1,linewidth=0.2,linestyle="dashed",label="1st seed")
    plt.plot(range,t2,marker="d",color='r',ms=1,linewidth=0.2,linestyle="dashed",label="2nd seed")
    plt.plot(range,t3,marker="d",color='g',ms=1,linewidth=0.2,linestyle="dashed",label="3rd seed")
    plt.plot(range,t4,marker="d",color='m',ms=1,linewidth=0.2,linestyle="dashed",label="4th seed")
    plt.plot(range,t5,marker="d",color='c',ms=1,linewidth=0.2,linestyle="dashed",label="5th seed")
    plt.plot(range,t6,marker="d",color='b',ms=1,linewidth=0.2,linestyle="dashed",label="6th seed")

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


def get_p(path):

    canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
    dim_t, dim_x = np.shape(canvas)
    t = 0
    R2 = 1.0

    while R2 >= 0.99:
        try:
            cross_section = list(np.abs(canvas[t-1,:])**2)
            space = list((np.array(range(len(cross_section)))  + 1 - (len(cross_section)/2))/2)
            cross_section = cross_section[::2]
            space = space[::2]
            R2, width = fit(space, cross_section)
            if not R2 >= 0.99:
                break 
            t -= 1
        except:
            break

    return (dim_t+t)/2


if __name__ == '__main__':
    main()