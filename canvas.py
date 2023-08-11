import numpy as np
import matplotlib.pyplot as plt
import math

T = 2.5
canvas = np.zeros((int(2*T), int(4*T)-2))
canvas[0,int(2*T)-1] = 1.0


for t in range(1, int(2*T)):
    for x in range(-t, t):
        print(t/2)
        print(-x/2)
        x_h = math.ceil(t/2 - x/2)
        print(x_h)
        x_v = math.floor(t/2 + 1 + x/2)
        print(x_v)
        print("\n")

        canvas[t,x+int(2*T)-1] = 0.5



plt.imshow(canvas,cmap='hot', interpolation='nearest')
plt.show()