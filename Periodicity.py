import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

q = 2
#try 1000 it's good!
strengths = np.arange(0,100,0.05)
Traces = [[],[],[],[],[],[]]
avg = []

for i in [1,2,3,4,5,6]:

    P_path = f"./Sample_Perturbation_{i}.csv"
    G = np.loadtxt(P_path,delimiter=',',dtype='complex_')
    G /= np.linalg.norm(G)
 
    for e in strengths:
        P = expm(1j*e*G)
        norm = np.linalg.norm(P - np.eye(q**4))
        Traces[i-1].append(norm)

for i in range(len(strengths)):
    avg.append((Traces[0][i]+Traces[1][i]+Traces[2][i]+Traces[3][i]+Traces[4][i]+Traces[5][i])/6)


fig, ax = plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("$\epsilon$", size=14, fontname="Times New Roman", labelpad=10)
plt.ylabel("|e$^{i{\epsilon}P}$ - I|$_F$" ,size=14, fontname="Times New Roman", labelpad=10)
plt.title("Trace Distance from Perturbation to Identity vs. Perturbation Strength", fontweight ='bold',size=14, fontname="Times New Roman")
ax.minorticks_on()
major_ticks = np.arange(strengths[0],strengths[-1],5)
minor_ticks = np.arange(strengths[0],strengths[-1],0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='minor', linewidth=0.5, alpha=0.5)

plt.plot(strengths,avg,color='k',linewidth=0.6,label="Average")
plt.plot(strengths,Traces[0],color='k',linewidth=0.2,linestyle="dashed",label="1st Seed")
plt.plot(strengths,Traces[1],color='r',linewidth=0.2,linestyle="dashed",label="2nd Seed")
plt.plot(strengths,Traces[2],color='g',linewidth=0.2,linestyle="dashed",label="3rd Seed")
plt.plot(strengths,Traces[3],color='b',linewidth=0.2,linestyle="dashed",label="4th Seed")
plt.plot(strengths,Traces[4],color='m',linewidth=0.2,linestyle="dashed",label="5th Seed")
plt.plot(strengths,Traces[5],color='c',linewidth=0.2,linestyle="dashed",label="6th Seed")

plt.grid()
plt.legend()
plt.show()