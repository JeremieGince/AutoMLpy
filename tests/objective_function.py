import numpy as np
import matplotlib.pyplot as plt

A = 3
x_0 = 0
sigma_x = 1
y_0 = 0
sigma_y = 1


def f(x, y):
    return A*np.sin(np.sqrt(((x - x_0) ** 2) / (2*sigma_x**2) + ((y - y_0) ** 2) / (2*sigma_y**2)))


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()