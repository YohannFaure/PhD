## Imports
# Science
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())

##

data_fig_4 = np.load(main_folder+"data_for_figures/figure_4.npy",allow_pickle=True).all()

lc_list_nuc                 = data_fig_4["lc_list_nuc"]
ell_per_lc_nuc              = data_fig_4["ell_per_lc_nuc"]
ell_per_lc_nuc_solid        = data_fig_4["ell_per_lc_nuc_solid"]
lc_solid_mean               = data_fig_4["lc_solid_mean"]
ell_per_lc_nuc_width        = data_fig_4["ell_per_lc_nuc_width"]
ell_per_lc_nuc_width_solid  = data_fig_4["ell_per_lc_nuc_width_solid"]



data_fig_5 = np.load(main_folder + "data_for_figures/figure_5.npy",allow_pickle=True).all()

lc_solid_mean                   = data_fig_5["lc_solid_mean"]
ell_per_lc_round_solid          = data_fig_5["ell_per_lc_round_solid"]
lc_list_round                   = data_fig_5["lc_list_round"]
ell_per_lc_round                = data_fig_5["ell_per_lc_round"]
ell_per_lc_round_width          = data_fig_5["ell_per_lc_round_width"]
ell_per_lc_round_width_solid    = data_fig_5["ell_per_lc_round_width_solid"]


mean_lc                         = data_fig_5["mean_lc"]
lc_event                        = data_fig_5["lc_event"]
ell                             = data_fig_5["ell"]
is_solid                        = data_fig_5["is_solid"]




###

fig,ax_4 = plt.subplots(1)
fig.set_size_inches(size_fig_ell_nuc_ell_slip)


ax_4.errorbar(ell_per_lc_nuc*1000/15,
        ell_per_lc_round*1000/30,
        xerr = ell_per_lc_nuc_width*1000/15,
        yerr = ell_per_lc_round_width*1000/30,
        fmt=" ",capsize=error_bar_width,color="k",
        ecolor=error_bar_color,
        elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_4.scatter(ell_per_lc_nuc*1000/15,
        ell_per_lc_round*1000/30,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=2,
        edgecolors="k",linewidth=0.01)


ax_4.errorbar(ell_per_lc_nuc_solid*1000/15,
        ell_per_lc_round_solid*1000/30,
        xerr = ell_per_lc_nuc_width_solid*1000/15,
        yerr = ell_per_lc_round_width_solid*1000/30,
        fmt=" ",capsize=error_bar_width,color="k",
        ecolor=error_bar_color,
        elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)


ax_4.scatter(ell_per_lc_nuc_solid*1000/15,
        ell_per_lc_round_solid*1000/30,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=2,
        edgecolors="k",linewidth=0.01)



ax_4.plot([0,4.3],[0,4.3],linestyle = '--',color = "k")


#ax_4.legend()
ax_4.set_xlim(0.8,4.3)
ax_4.set_ylim(0.8,4.3)
#ax_4.set_ylim(23,120)
ax_4.set_xlabel(r"$\frac{\left\langle d_{nuc}\right\rangle}{\ell_{eye}\,/\,2}$",labelpad = 1.5)
ax_4.set_ylabel(r"$\left\langle\ell_{slip}\right\rangle\,/\,\ell_{eye}$")

ax_4.xaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.xaxis.set_major_locator(MultipleLocator(1))
ax_4.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.yaxis.set_major_locator(MultipleLocator(1))
#ax_4.yaxis.set_minor_locator(MultipleLocator(25))
#ax_4.yaxis.set_major_locator(MultipleLocator(50))


set_grid(ax_4)
ax_4.set_aspect('equal')

plt.tight_layout()


fig.savefig(main_folder + "Figures_chap_article/ell_nuc_ell_slip.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_article/ell_nuc_ell_slip.pdf")
fig.savefig(main_folder + "Figures_chap_article/ell_nuc_ell_slip.svg")
plt.close('all')


