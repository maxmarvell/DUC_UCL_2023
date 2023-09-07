from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objects as go
import pandas as pd
import numpy as np

matrix = np.loadtxt('data/PathIntegralTruncated/2_4146150358_1e-07.csv',delimiter=',',dtype='complex_',skiprows=1)
matrix = np.abs(matrix[::2,::2])

t = [float(i) for i in range(0,matrix.shape[0])]
x = [float(i) for i in range(-matrix.shape[0]+1,matrix.shape[0])]

surface_plot = go.Surface(
        x=x, 
        y=t,
        z=matrix,
        # coloraxis = 'coloraxis',
        opacity = 0.99,
        colorscale = 'Hot',
        showscale = True
    )

matrix = np.loadtxt('data/TensorExact/2_3806443768_3e-07.csv',delimiter=',',dtype='complex_',skiprows=1)
matrix = np.abs(matrix[::2,::2])

t = [float(i) for i in range(0,matrix.shape[0])]
x = [float(i) for i in range(-matrix.shape[0]+1,matrix.shape[0])]

surface_plot_2 = go.Surface(
        x=x, 
        y=t,
        z=matrix,
        # coloraxis = 'coloraxis',
        opacity = 0.3,
        colorscale = 'Blues',
        showscale = False
    )

fig = go.Figure(data = [surface_plot,surface_plot_2])
fig.show()