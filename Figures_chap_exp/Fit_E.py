## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import interpolate
# random
import threading
import pickle


main_folder = 'E:/manuscrit_these/'
main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


###

dico = np.load( main_folder + "data_for_figures/dicto_fit_E_(2022-11-09).npy",
                allow_pickle = True).all()

locals().update(dico)





### Compute E and nu
S=607e-6
E,Eerr=np.polyfit(eps_yy,Fn/S,1)

nu_x_1=np.polyfit(Fn,eps_xx_1*E*S,1)[0]
nu_z_1=np.polyfit(Fn,eps_zz_1*E*S,1)[0]
nu_x_2=np.polyfit(Fn,eps_xx_2*E*S,1)[0]
nu_z_2=np.polyfit(Fn,eps_zz_2*E*S,1)[0]

nu_x=(nu_x_1+nu_x_2)/2
nu_z=(nu_z_1+nu_z_2)/2

nu_perp=np.polyfit(Fn,-(eps_xx+eps_zz)*E*S/2,1)[0]

eps_perp=(eps_xx+eps_zz)/2




### Plot

fig, axs = plt.subplots(1,2,sharex=True,sharey=True)


axs[0].scatter(Fn,-1000*eps_yy_1,c=secondary_plot_color,alpha=1,s=10,marker="x",label=r"$\epsilon_{yy}^{\{1,2\}}$")
axs[0].scatter(Fn,-1000*eps_yy_2,c=secondary_plot_color,alpha=1,s=10,marker="x")
axs[0].scatter(Fn,-1000*eps_yy,c=main_plot_color,alpha=1,s=10,marker="x",label=r"$\left\langle\epsilon_{yy}\right\rangle$")
axs[0].plot(Fn,-1000*(Fn/S-Eerr)/E,c="r", label=r"ajust., $ E={:.2f} $".format(-E/1e9)+r" GPa",linewidth=2)
axs[0].set_xlabel("$F_N$ (N)")
axs[0].set_ylabel("$\epsilon_{yy}$ (mm/m)")
# legend=axs[0].legend()
# legend.get_frame().set_alpha(1)
# legend.get_frame().set_facecolor((1, 1, 1, 1))

handles, labels = axs[0].get_legend_handles_labels()
order = [2,1,0]
axs[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order],framealpha = 1, facecolor = (1,1,1,1),loc = "upper left",bbox_to_anchor=(0,1),handletextpad=0.5,handlelength =1,prop={'size': 9})





axs[0].grid(which="both")
axs[0].yaxis.set_major_locator(MultipleLocator(1))
axs[0].yaxis.set_minor_locator(MultipleLocator(.5))
axs[0].xaxis.set_major_locator(MultipleLocator(2500))
axs[0].xaxis.set_minor_locator(MultipleLocator(1250))
axs[0].set_xlim([0,5000])


#axs[1].scatter(Fn/10,1000*eps_xx,c=secondary_plot_color,alpha=1,s=10,marker="x",label=r"$\epsilon_{\parallel}^{(avg)}$")
#axs[1].scatter(Fn/10,1000*eps_zz,c=secondary_plot_color,alpha=1,s=10,marker="x")
axs[1].scatter(Fn,1000*(eps_xx+eps_zz)/2,c=main_plot_color,alpha=1,s=10,marker="x",label=r"$\left\langle\epsilon_{\perp}\right\rangle$")
axs[1].plot(Fn,-nu_perp*1000*Fn/(S*E),c="r", label=r"ajust., $\nu={:.2f}$".format(nu_perp),linewidth=2)
axs[1].set_xlabel("$F_N$ (N)")
axs[1].set_ylabel("$\epsilon_{\perp}$ (mm/m)")

axs[1].grid(which="both")


handles, labels = axs[1].get_legend_handles_labels()
order = [1,0]
axs[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order],framealpha = 1, facecolor = (1,1,1,1),loc = "upper left",bbox_to_anchor=(0,1),handletextpad=0.5,handlelength =1,prop={'size': 9})


set_grid(axs)

fig.set_size_inches(size_fig_Fit_E)
plt.tight_layout()
fig.subplots_adjust(bottom=0.16, top=0.98)

plt.savefig(main_folder + "Figures_chap_exp/fit_E.png",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/fit_E.pdf",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/fit_E.svg",dpi=600)
plt.close()


