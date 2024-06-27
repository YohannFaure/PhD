
main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())



import matplotlib.patches as patches

roll_smooth=20
start=26000
end=-100000


def smooth(a, n=10) :
    """
    A simple denoising trick using rolling average.
    A more elaborate harmonic filtering could be useful.
    """
    ret = np.cumsum(a, axis=-1, dtype=float)
    ret[...,n:] = ret[...,n:] - ret[...,:-n]
    return( ret[...,n - 1:] / n )




fig,ax=plt.subplots(1)
fig.set_size_inches(size_fig_fully_gran)
loc = main_folder + "data_for_figures/200kg_stopper_daq.txt"
data=np.loadtxt(loc)

start=44000
end=-17000
data=smooth(data,roll_smooth)


f=1000


time=np.arange(data.shape[-1])/f


n=1

ax.plot(time[start:end]-time[start],(-data[n][start:end]+data[n][start])/np.mean(data[0]),label=None,c="limegreen")
ax.grid(which="both")
ax.set_xlabel("temps (s)")
ax.set_ylabel("$\mu=F_S\,/\,F_N$")





ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.1))


xxx=1.4

patch = patches.Circle((145/xxx, 125/xxx), 15, alpha=0.3, transform=None, color="r")
fig.artists.append(patch)

patch2 = patches.Circle((240/xxx, 180/xxx), 15, alpha=0.3, transform=None,color="b")
fig.artists.append(patch2)

#ax.legend()


ax.set_xlim(0,365)
ax.set_ylim(0,0.4)
plt.tight_layout()

set_grid(ax)

plt.savefig(main_folder + "Figures_chap_article/fully_gran.svg")
plt.savefig(main_folder + "Figures_chap_article/fully_gran.pdf")
plt.savefig(main_folder + "Figures_chap_article/fully_gran.png",dpi=dpi_global)
plt.close('all')




