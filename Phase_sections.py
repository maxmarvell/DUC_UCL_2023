import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


path = "./T24_d3_e0.0000004"
canvas = np.loadtxt(path,delimiter=',',dtype='complex_')
print(np.shape(canvas))
dim_t, dim_x = np.shape(canvas)
T = 24.0
e = 0.0000004


space = list((np.array(range(dim_x))  + 1 - (dim_x/2))/2)
space = space[::2]
fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("x", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("arg(<Z$_x$|Z(t)>)" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title(f"Spatial Distribution of arg(<Z$_x$|Z(t)>) \n $\epsilon$ = {e}", fontweight ='bold',size=14, fontname="Times New Roman")

cross_section = list(np.angle(canvas[-1,:]))
cross_section = cross_section[::2]
plt.scatter(space, cross_section, marker="d", color='k', s=5)


ax.minorticks_on()
major_ticks = np.arange(space[0],space[-1],2)
minor_ticks = np.arange(space[0],space[-1],0.25)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)
plt.grid()
plt.legend()
plt.show()
