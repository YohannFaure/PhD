import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


def smooth(a, n=10) :
    """
    A simple denoising trick using rolling average.
    A more elaborate harmonic filtering could be useful.
    """
    ret = np.cumsum(a, axis=-1, dtype=float)
    ret[...,n:] = ret[...,n:] - ret[...,:-n]
    return( ret[...,n - 1:] / n )

n_smooth=10

#test = scio.loadmat("E:/2023-2024/2024-01-09-extraction-profilo-old-data/two_profiles.mat")
test = scio.loadmat(main_folder + "data_for_figures/two_profiles.mat")

x = test["x"].flatten()[:-2].astype(float)
y1 = test["y"].flatten()[:-2]
y2 = test["y2"].flatten()

x=x*150/len(x)

fig=plt.figure()
fig.set_size_inches(size_fig_profil_bloc)

y= smooth((y1+y2)/2,n_smooth)+5

plt.plot(smooth(x,n_smooth), y-1.1 ,c=main_plot_color)



plt.xlabel("position $x$ (mm)")
plt.ylabel("profil d'altitude $y$ (µm)")
plt.ylim([-2,17])
plt.xlim([-5,155])

ax=plt.gca()
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.xaxis.set_major_locator(MultipleLocator(50))
set_grid(ax)

plt.tight_layout()
plt.savefig(main_folder + "Figures_chap_exp/profil_bloc.png",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp//profil_bloc.pdf",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/profil_bloc.svg",dpi=600)
plt.close()









