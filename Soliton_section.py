import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


path = "./Larders/Larder_5/T24.0_d3_e5"
canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
dim_t, dim_x = np.shape(canvas)
T = 24.0
soliton = []
time = np.arange(0.0,24.0,0.5)

for t in range(int(2*T)):
    x = int(2*T)-t-1
    soliton.append((np.abs(canvas)**2)[t,x])


fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("t", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("|<Z$_{-t}$|Z(t)>|$^2$" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title("Decay of Soliton along the Light Cone", fontweight ='bold',size=14, fontname="Times New Roman")
ax.minorticks_on()
major_ticks = np.arange(time[0],time[-1],10)
minor_ticks = np.arange(time[0],time[-1],0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)

plt.plot(time,soliton,marker="d",color='k',ms=5,linewidth=0.3,linestyle="dashed",label="$\epsilon$=10")

plt.grid()
plt.legend()
plt.show()