main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())

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




fig,axes=plt.subplots(nrows=1,ncols=2,figsize=size_fig_fullygran ,sharex=False,sharey=True)

loc = main_folder + "data_for_figures/2021-04-01/100kg_stopper_daq.txt"
data=np.loadtxt(loc)


data=smooth(data,roll_smooth)


f=1000


time=np.arange(data.shape[-1])/f


n=1


axes[0].plot(time[start:end]-time[start],(-data[n][start:end]+data[n][start])/np.mean(data[0]),label="$F_N={}$ N".format(10 * int(np.mean(data[0]))),c=granular_color)
axes[0].set_ylabel("$\mu$")
axes[0].set_yticks([0,0.1,0.2,0.3])
axes[0].set_xlabel("temps (s)")
axes[0].legend()

loc = main_folder + "data_for_figures/2021-04-01/200kg_stopper_daq.txt"
data=np.loadtxt(loc)

start=44000
end=-17000
data=smooth(data,roll_smooth)


f=1000


time=np.arange(data.shape[-1])/f


n=1

axes[1].plot(time[start:end]-time[start],(-data[n][start:end]+data[n][start])/np.mean(data[0]),label="$F_N={}$ N".format(10 * int(np.mean(data[0]))),c=granular_color)
axes[1].set_xlabel("temps (s)")
axes[1].set_yticks([0,0.1,0.2,0.3])
axes[1].legend(facecolor = (1,1,1,1),framealpha=1)

axes[1].set_xlim([0,370])
axes[0].set_xlim([0,380])
axes[1].set_ylim([0,0.37])

set_grid(axes)

plt.tight_layout()
plt.savefig(main_folder+ "Figures_chap_conclusion/fullygran.pdf")
plt.savefig(main_folder+ "Figures_chap_conclusion/fullygran.png")
plt.savefig(main_folder+ "Figures_chap_conclusion/fullygran.svg")
plt.close()





