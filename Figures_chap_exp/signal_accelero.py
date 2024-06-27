import numpy as np
import matplotlib.pyplot as plt

main_folder = 'E:/manuscrit_these/'
main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


time = np.arange(-10,10,0.001)
short_time=np.arange(0,5,0.001)

def exp_sin_posit(x,omega,tau):
    signal = np.sin(x*omega)*np.exp(-x/tau)
    signal = signal*(signal>0)
    return(signal)


omega=20
tau=0.5
amp=0.01

variation = 0.8

signal_tot = np.zeros_like(time)
long_pulse = np.zeros_like(time)
short_pulse = np.zeros_like(time)

starts = [1000,10000,12000]

for i in starts:
    signal =  exp_sin_posit(short_time, omega*(0.8+variation*np.random.random(1)),tau*(0.5+variation*np.random.random(1)))
    signal_tot[i:i+len(short_time)]=signal
    if long_pulse[i]==0:
        short_pulse[i:i+500]=1
    long_pulse[i:i+3000]=1


signal_tot += np.random.random(signal_tot.shape)*amp



fig,ax = plt.subplots(1)
fig.set_size_inches(size_signal_acc)


plt.plot(time,3*signal_tot,label="accéléromètre",alpha=0.8,linewidth=main_linewidth,zorder=4,c="royalblue")
plt.plot(time,5*long_pulse,label=r"$1^{\text{er}}$ monostable",alpha=0.8,linewidth=main_linewidth*2,zorder=2,c="peru")
plt.plot(time,5*short_pulse,label=r"$2^{\text{ème}}$ monostable",alpha=0.8,linewidth=main_linewidth,zorder=3,c="forestgreen")
plt.ylim([-0.5,5.5])
plt.xlim([-10,10])
plt.xticks([])

ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))


plt.ylabel("signal (V)")
plt.xlabel("temps (u.a.)")


plt.legend(facecolor = (1,1,1,1),framealpha=1)
plt.tight_layout()
set_grid(ax)
fig.text(0,0.93,"b.",fontweight="bold",size=12)


plt.savefig(main_folder + "Figures_chap_exp/signal_accelero.png",dpi=1200)
plt.savefig(main_folder + "Figures_chap_exp/signal_accelero.pdf")
plt.savefig(main_folder + "Figures_chap_exp/signal_accelero.svg")
plt.close()


