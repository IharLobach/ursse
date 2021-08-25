import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.style.use('seaborn-pastel')
from mpl_toolkits.axisartist.axislines import SubplotZero
plt.rcParams.update({
    "text.usetex": True})


fig = plt.figure()
# ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
beta = 1
e = 1
alpha = -0.5
psis = np.linspace(0, 2*np.pi, 100)
def xf(psi):
    return np.sqrt(e*beta)*np.cos(psi)
def yf(psi):
    return -np.sqrt(e/beta)*(np.sin(psi)+alpha*np.cos(psi))
x = xf(psis)
y = yf(psis)
ax.plot(x, y, lw=3)
psi0 = [0]
dpsi = 0.2
xs = [xf(psi0[-1])]
ys = [yf(psi0[-1])]
line, = ax.plot([], [], 'o')
for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
    ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])

ax.annotate("$x$", (ax.get_xlim()[1],
                    ax.get_ylim()[1] * 0.05),
            fontsize=30)

ax.annotate("$x'$", (ax.get_xlim()[1] * 0.05,
                    ax.get_ylim()[1]),
            fontsize=30)



def init():
    line.set_data(xs, ys)
    return line,
def animate(i):
    psi0.append(psi0[-1] + dpsi)
    xs.append(xf(psi0[-1]))
    ys.append(yf(psi0[-1]))
    line.set_data(xs, ys)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=50, blit=True)


anim.save('betatron_motion.gif', writer='imagemagick')
