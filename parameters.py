import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import scipy.io as scio
from scipy import interpolate
import os
import sys

#main_folder = 'E:/manuscrit_these/'


#sys.path.insert(0, main_folder+"DAQ_Python")


#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preview'] = True
#plt.rc('font', family='serif')

plt.rcParams['font.family'] = 'Arial'


# chose colors


main_plot_color = "dodgerblue"
secondary_plot_color = "lightsteelblue"

error_bar_color="grey"
error_bar_alpha=1
dashed_color="k"
secondary_plot_alpha = 0.5

solid_solid_color = "navy"
granular_color = "peru"
solid_in_granular_color="royalblue"
hole_in_solid_color="navy"



# chose sizes and shapes



# chose sizes and shapes
solid_marker = "o"
scatter_size = 15
scatter_marker = "d"
error_bar_width=1.5
markeredgewidth=.5
dash_line_style = ":"

SMALL_SIZE = 9
MEDIUM_SIZE = 10
LARGE_SIZE = 11


INSET_SMALL_SIZE = 7
INSET_MEDIUM_SIZE = 8
INSET_LARGE_SIZE = 9
inset_label_pad = 1


LETTER_SIZE=12





# chose names
sliding_perc_name_full = r"Glissement inter-évènement normalisé"
sliding_perc_name_short = r"$S=\delta_{IE}\,/\,\delta_{tot}$"

sliding_ie_name_full = r"Glissement inter-évènement (mm)"
sliding_ie_name_short = r"$\delta_{IE}$ (mm)"

hists_y_label = r"$N_{ev}\,/\,N_{tot}$"
hists_y_label_2 = r"$N_{SL}\,/\,N_{ev}$"

freq_name_full = "Fréquence du stick-slip (Hz)"
#freq_name_short = r"$\left< f_{ss} \right>$ (Hz)"
freq_name_short = r"$\left< 1\,/\,\Delta T \right>$ (Hz)"

tot_sliding_full = "Glissement total (mm)"
tot_sliding_short = "$\delta_{tot}$ (mm)"

LC_name_long = "Contraste de chargement"
LC_name_short = r"C_{\sigma}"


dpi_global=1200








# chose line width
main_linewidth=1


# set up grid
grid_major_color="darkgray"
grid_major_width=.5
grid_major_lines="-"
grid_major_alpha=.7
grid_minor_color="lightgray"
grid_minor_width=.3
grid_minor_lines="-"
grid_minor_alpha=.5

mm=1/25.4

maxwidth = 170*mm
# Sizes (w,h)
size_fig_Fit_E = (140*mm, 65*mm)
size_fig_profil_bloc = (100*mm, 80*mm)
size_test_JPBox = (140*mm, 80*mm)
size_signal_acc = (150*mm, 50*mm)
size_correl_0 = (85*mm, 38*mm)
size_correl_1 = (170*mm, 28*mm)
size_correl_2 = (130*mm, 50*mm)
size_tracking = (130*mm, 60*mm)
size_resolution_tracking_synthetic = (170*mm,120*mm)
size_fig_S2 = (170*mm, 140*mm)
size_fig_S2_res_only = (160*mm, 65*mm)

size_fig_article_1 = (160*mm, 90*mm)
size_fig_article_1d = (90*mm, 70*mm)
size_fig_article_2 = (120*mm, 100*mm)
size_fig_article_3 = (110*mm, 70*mm)
size_fig_article_4 = (170*mm, 160*mm)
size_fig_article_4_eh = (170*mm, 100*mm)
size_fig_article_6 = (170*mm, 60*mm)
size_fig_article_6_backup = (120*mm, 60*mm)
size_fig_creep_5_paires = (170*mm, 50*mm)
size_fig_ss_vs_eh = (140*mm,60*mm)
size_fig_lc_freq = (60*mm,60*mm)
size_fig_fully_gran = (90*mm,60*mm)
size_fig_profiloplot_zoom = (100*mm,80*mm)
size_fig_yield = (90*mm,60*mm)
size_fig_ell_nuc_ell_slip = (60*mm,60*mm)
size_fig_ell_triple = (170*mm, 50*mm)
size_fig_ell_vs_delta_fs = (60*mm, 60*mm)
size_fig_LEFM = (170*mm, 60*mm)

size_fig_fullygran = (120*mm, 60*mm)


x_plot=(np.array([1,2,3,4,5,8,11,12,13,14])-0.5)*10
bins=np.array([-1,-0.5,0,1,2])
pad_title_hists=-7


size_fig_1 = (186*mm, 90*mm)

# # # #


def set_grid(axes):
    if type(axes)==list:
        axes=np.array(axes)
    if type(axes)==np.ndarray:
        for ax in axes.flatten():
            ax.set_facecolor('none')
            ax.set_axisbelow(True)
            ax.grid(visible=True,which='major', color=grid_major_color, linestyle=grid_major_lines, linewidth=grid_major_width,alpha=grid_major_alpha)
            ax.grid(visible=True,which='minor', color=grid_minor_color, linestyle=grid_minor_lines, linewidth=grid_minor_width,alpha=grid_minor_alpha)
    else:
        axes.set_facecolor('white')
        axes.grid(visible=True,which='major', color=grid_major_color, linestyle=grid_major_lines, linewidth=grid_major_width,alpha=grid_major_alpha)
        axes.grid(visible=True,which='minor', color=grid_minor_color, linestyle=grid_minor_lines, linewidth=grid_minor_width,alpha=grid_minor_alpha)




mpl.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
mpl.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title
mpl.rcParams['lines.linewidth'] = main_linewidth
mpl.rcParams['axes.facecolor'] = 'none'
mpl.rcParams['axes.linewidth'] = .5
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['xtick.minor.width'] = .3
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['ytick.minor.width'] = .3
mpl.rcParams['xtick.major.size'] = 2.5
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['ytick.major.size'] = 2.5
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['xtick.major.pad']='1'
mpl.rcParams['ytick.major.pad']='1'




def set_up_inset(axin,linewidth=.5):
    axin.tick_params(labelsize=INSET_SMALL_SIZE)
    axin.set_xlabel(axin.get_xlabel(),fontsize=INSET_MEDIUM_SIZE)
    axin.set_ylabel(axin.get_ylabel(),fontsize=INSET_MEDIUM_SIZE)
    if not ((linewidth is None) or (linewidth == 0.)):
        for line in axin.lines:
            line.set_linewidth(linewidth)
    axin.tick_params(axis='y', pad=1)
    axin.tick_params(axis='x', pad=1)

def smooth(a, n=10) :
    """
    A simple denoising trick using rolling average.
    A more elaborate harmonic filtering could be useful.
    """
    ret = np.cumsum(a, axis=-1, dtype=float)
    ret[...,n:] = ret[...,n:] - ret[...,:-n]
    return( ret[...,n - 1:] / n )

def real_tight_layout(fig):
    fig.tight_layout()
    fig.align_labels()
    left, right, bottom, top, wspace, hspace = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.bottom, fig.subplotpars.top, fig.subplotpars.wspace, fig.subplotpars.hspace
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return(left, right, bottom, top, wspace, hspace)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx




