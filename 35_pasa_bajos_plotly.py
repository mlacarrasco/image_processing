
# miguel carrasco
# version 1.0

import plotly.graph_objects as go
import numpy as np


x = np.linspace(-127, 128,128)
y = np.linspace(-127, 128,128)

# generamos una malla con puntos
X, Y = np.meshgrid(x,y)
H = np.exp(-0.0005*(np.power(X,2)+np.power(Y,2)))

# dejamos los datos en una sola dimension
x = X.flatten()
y = Y.flatten()
z = H.flatten()


mesh = go.Mesh3d(x=x, y=y, z=z, color='lightpink',opacity=0.50)
line_marker = dict(color='blue', width=0.5)

fig = go.Figure(data=[mesh])

for xx, yy, zz in zip(X, Y, H):
    fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines',line=line_marker, opacity=0.2,name='')

fig.update_layout(width=700, height=700, showlegend=False)

fig.show()