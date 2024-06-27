###
main_folder = 'E:/manuscrit_these/'
main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())

from matplotlib import cm
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


X,Y,Z = np.load(main_folder+"data_for_figures/dataprofilo.npy")

Z1 = gaussian_filter(Z, sigma=2)
Z2 = gaussian_filter(Z, sigma=3)

Z1-=np.min(Z1)

Z1=-Z1

X-=np.min(X)
Y-=np.min(Y)
X*=100/np.max(X)
Y*=100/np.max(Y)

fig,ax = plt.subplots(1)
fig.set_size_inches(size_fig_profiloplot_zoom )

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.15)

im = ax.imshow(Z1, cmap=cm.coolwarm.reversed(), extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)])

fig.colorbar(im, cax=cax, orientation='vertical',label='y (µm)', ticks=[-6,-4,-2,0])

ax.set_xticks([0,50,100])
ax.set_yticks([0,50,100])
ax.set_xlabel('$x$ (µm)')
ax.set_ylabel('$z$ (µm)')
fig.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=1)



plt.savefig(main_folder + "Figures_chap_intro/profiloplot_zoom.png",dpi=1200)
plt.savefig(main_folder + "Figures_chap_intro/profiloplot_zoom.pdf",dpi=1200)
plt.savefig(main_folder + "Figures_chap_intro/profiloplot_zoom.svg",dpi=1200)
plt.close('all')
##

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm)

# Add a color bar which maps values to colors
fig.colorbar(surf)

plt.show()
"""
