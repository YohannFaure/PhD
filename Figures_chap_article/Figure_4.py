## Imports
# Science
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())






loaded_data = np.load(main_folder + "data_for_figures/figure_4.npy",allow_pickle=True).all()
locals().update(loaded_data)



########################## create figure



# nested gridspecs, yeaaaaah...
fig = plt.figure()
fig.set_size_inches(size_fig_article_4)

gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios = [9,3],
                      left=0.07, right=0.97,
                      top=0.97,bottom=0.07,
                      hspace=0.22, wspace=0)

#nrows=nrows, ncols=11, left=0.06, right=0.97,
#                      top=0.98,bottom=0.06,
#                      hspace=0, wspace=0

gs00 = gridspec.GridSpecFromSubplotSpec(10, 3, subplot_spec=gs0[0],
                      width_ratios = [5,0.8,5],
                      hspace=0, wspace=0)

axes_1=[]
axes_3=[]

for i in range(10):
    axes_1.append( fig.add_subplot(gs00[i,0]) )
    axes_1[-1].sharex(axes_1[0])
    axes_1[-1].sharey(axes_1[0])

    axes_3.append( fig.add_subplot(gs00[i, -1]) )
    axes_3[-1].sharex(axes_3[0])
    axes_3[-1].sharey(axes_1[0])




gs01 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec = gs0[1],
                      width_ratios = [1,1,1,1,1,0.6,2],
                      hspace=0, wspace=0)
axes_2=[]
for i in range(5):
    axes_2.append( fig.add_subplot(gs01[0, i]) )
    axes_2[-1].sharex(axes_2[0])
    axes_2[-1].sharey(axes_2[0])


gs02 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gs01[-1],
                      height_ratios = [0.5,1],
                      hspace=0, wspace=0)

ax_4 = fig.add_subplot(gs02[-1])
ax_drawing = fig.add_subplot(gs02[0])








########################## Data for propagation


# control smoothing of the data (rolling average) and starting point
roll_smooth=5
start=0
n_plot=10

## Location of the data inside the file
# channels containing the actual strain gages
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])
# channels containing the normal and tangetial force
forces_channels = [32,33]
# channel containing the trigger
trigger_channel = 34

## Parameters
# default x, just in case there's nothing in the saved params
x=np.array([1,2,3,4,5,8,11,12,13,14])*1e-2 -0.005
# Load the params file, and extract frequency of acquisition







## plot data
# lines propag
xs=np.array([0.17,0.155,0.215])*1.05
ys=np.array([0.7,0.775,0.97])*0.992
line = mpl.lines.Line2D(xs, ys, lw=1, color='r', alpha=1)
fig.add_artist(line)

xs=np.array([0.31,0.35])*1.03
ys=np.array([0.6,0.4])*0.992
line = mpl.lines.Line2D(xs, ys, lw=1, color='r', alpha=1)
fig.add_artist(line)



color=[solid_in_granular_color]*5+[granular_color]+[solid_in_granular_color]*4

for i in range(10):
    axes_1[i].plot(1000*fast_time[start:]-1000*fast_time.mean(),1000*(gages[3*(i+1)-1][start:]-gages[3*(i+1)-1][0:10000].mean()),color=color[i])
    axes_1[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes_1[i].grid("both")
    # dashes
    axes_1[i].axvline(x=indexes[i]*1000-1000*fast_time.mean(),c="k",linestyle="--")

axes_1[3].scatter(indexes[3]*1000-1000*fast_time.mean(), 0, s=50, marker='*', color='r', zorder=3)
fig.text(0.10, 0.87, r"$v\sim 800$ m/s", size = INSET_SMALL_SIZE, rotation =0, c="r")
fig.text(0.34, 0.51, r"$v\sim 1000$ m/s", size = INSET_SMALL_SIZE, rotation = 0,c="r")
#txt1.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))



axes_1[-1].set_xlabel('temps (ms)')
axes_1[4].set_ylabel(r'$\varepsilon_{xy}-\varepsilon_{xy}^0$ (mm/m)',size=MEDIUM_SIZE)


# adjust ticks
axes_1[-1].set_xlim([-0.16,0.06])
axes_1[-1].set_xticks([-0.15,-0.1,-0.05,0,0.05])
axes_1[-1].set_xticklabels(["-$0.15$","-$0.1$","-$0.05$","$0$","$0.05$"])
axes_1[-1].set_ylim([-0.19,0.06])
axes_1[-1].set_yticks([-0.1,0])
axes_1[-1].set_yticklabels(["$-0.1$","$0$"])



for i in range(9):
    plt.setp(axes_1[i].get_yticklabels(), visible=False)
    plt.setp(axes_1[i].get_xticklabels(), visible=False)




labels = [r"$x=5 \pm 0.25$ mm"]+[r"$x={}$".format(i) for i in [int(x_plot[i]) for i in range(1,len(x_plot))]]

# Iterate through each subplot and add data
for i, ax in enumerate(axes_1):
    ax.annotate(labels[i], xy=(0.02, 0.1), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)





mean_lc=data_loaded["mean_lc"]
manip_num=np.array(data_loaded["manip_num"])
LC_here = mean_lc[manip_num==chosen_manip_fig41][0]


axes_1[0].annotate("${}=${:.2f}".format(LC_name_short,LC_here), xy=(0.75, 0.8), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)



set_grid(axes_1)



for i in range(1,5):
    plt.setp(axes_2[i].get_yticklabels(), visible=False)
    plt.setp(axes_2[i].get_yticklabels(), visible=False)



########################## Data for starting point

## Nice plot
def temp_nice_plot(y_min,x, ax = None ,ylim=None) :
    """
    plots a nice representation of the bloc in an histogram
    Useful way later
    """
    if ax is None:
        ax=plt.gca()
    if ylim is None:
        ylim=ax.get_ylim()
    else:
        ylim=ylim
    dilat = (ylim[1]-ylim[0])*0.7
    ax.set_ylim((y_min-0.1*dilat,ylim[1]))
    import matplotlib.patches as patches
    arc_radius = 0.05*dilat
    arc_center_x = 75
    arc_center_y = y_min
    start_angle = 0
    end_angle = 180
    arc_patch = patches.Arc((arc_center_x, arc_center_y), width=30, height=2*arc_radius, angle=0,
                            theta1=start_angle, theta2=end_angle, edgecolor=secondary_plot_color, linewidth=1)
    ax.add_patch(arc_patch)

    # lines of the block
    line1 = patches.ConnectionPatch((0, y_min), (60, y_min), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line1)
    line2 = patches.ConnectionPatch((90, y_min), (150, y_min), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line2)
    line3 = patches.ConnectionPatch((0, y_min), (0, y_min+0.25*dilat), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line3)
    line4 = patches.ConnectionPatch((150, y_min), (150, y_min+0.25*dilat), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line4)

    # gages, used and unused
    width_patch=(x[1]-x[0])/50
    for xi in x:
        if xi != 75.:
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.07*dilat), 2*width_patch, 0.01*dilat, color=solid_in_granular_color,alpha=1)
        else:
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.07*dilat), 2*width_patch, 0.01*dilat, color=granular_color,alpha=1)
        ax.add_patch(square_patch)




def temp_nice_plot_large(y_min,x, ax = None ,ylim=None) :
    """
    plots a nice representation of the bloc in an histogram
    Useful way later
    """
    if ax is None:
        ax=plt.gca()
    if ylim is None:
        ylim=ax.get_ylim()
    else:
        ylim=ylim
    dilat = (ylim[1]-ylim[0])*0.7
    ax.set_ylim((y_min-0.1*dilat,ylim[1]))
    import matplotlib.patches as patches
    arc_radius = 0.05*dilat
    arc_center_x = 75
    arc_center_y = y_min
    start_angle = 0
    end_angle = 180
    arc_patch = patches.Arc((arc_center_x, arc_center_y), width=30, height=3*arc_radius, angle=0,
                            theta1=start_angle, theta2=end_angle, edgecolor=secondary_plot_color, linewidth=1)
    ax.add_patch(arc_patch)

    # lines of the block
    line1 = patches.ConnectionPatch((0, y_min), (60, y_min), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line1)
    line2 = patches.ConnectionPatch((90, y_min), (150, y_min), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line2)
    line3 = patches.ConnectionPatch((0, y_min), (0, y_min+0.25*dilat), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line3)
    line4 = patches.ConnectionPatch((150, y_min), (150, y_min+0.25*dilat), "data", "data", edgecolor=secondary_plot_color, linewidth=1, arrowstyle="-")
    ax.add_patch(line4)

    # gages, used and unused
    width_patch=(x[1]-x[0])/10
    for xi in x:
        if xi != 75.:
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.15*dilat), 2*width_patch, 0.03*dilat, color=solid_in_granular_color,alpha=1)
        else:
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.15*dilat), 2*width_patch, 0.03*dilat, color=granular_color,alpha=1)
        ax.add_patch(square_patch)




## Load all data




index_start= np.array(data_loaded_3["start_per_event"])
loading_contrast = np.array(data_loaded_3["lc_per_event"])
from_solid=np.array(data_loaded_3["solid_per_event"])




## Define histograms

n_bin=len(bins)-1

def make_hists(bins, bin_variable, to_count):
    hists = []
    for i in range(n_bin):
        hists.append([([to_count[k]
                            for k in range(len(to_count))
                            if  bin_variable[k]<bins[i+1]
                            and bin_variable[k]>=bins[i]
                        ]).count(j) for j in range(10)])
    hists=np.array(hists)
    return(hists)


histogram_start_solid = np.array([list(index_start[from_solid]).count(i) for i in range(10)])
histograms_start=make_hists(bins, loading_contrast[np.logical_not(from_solid)], index_start[np.logical_not(from_solid)])





## Plot data axis 2

hist_y_lim=[-0.08,1]

y_min=-0.05
pad_title = pad_title_hists
y_title = 1+1e-50
width=0.8*min(np.diff(x_plot))


meanstarts_left  =[]
meanstarts_right =[]

for i in range(5):
    ax=axes_2[i]
    if i==0:
        ax.bar(x_plot, histogram_start_solid/histogram_start_solid.sum(),
                width=width,
                color = solid_solid_color,
                linewidth=.1,edgecolor='k')
        ax.set_title("à vide", pad=pad_title,y=y_title,size=INSET_MEDIUM_SIZE)
        temp_nice_plot(y_min,x_plot,ax=ax,ylim=None)
        ax.grid(True,which="both")
        meanstart = (x_plot*histogram_start_solid).sum()/histogram_start_solid.sum()
        meanstarts_left.append(meanstart)
        meanstarts_right.append(meanstart)
        ax.axvline(x=meanstart, ymax=0.8,linestyle='--',c='k')

    else:
        ax.bar(x_plot, histograms_start[i-1]/histograms_start[i-1].sum(),
                width=width,
                color = main_plot_color,linewidth=.1,edgecolor='k')
        ax.set_title(r"{}$\leq${}<{}".format(bins[i-1],r"${}$".format(LC_name_short),bins[i]), pad=pad_title, y=y_title,size=INSET_MEDIUM_SIZE)
        temp_nice_plot(y_min,x_plot,ax=ax,ylim=None)
        ax.grid(True,which="both")
        meanstart_left = (x_plot*histograms_start[i-1])[x_plot>75.].sum()/histograms_start[i-1][x_plot>75.].sum()
        meanstart_right= (x_plot*histograms_start[i-1])[x_plot<75.].sum()/histograms_start[i-1][x_plot<75].sum()
        ax.axvline(x=meanstart_left, ymax=0.8,linestyle='--',c='k')
        ax.axvline(x=meanstart_right, ymax=0.8,linestyle='--',c='k')
        meanstarts_left.append(meanstart_left)
        meanstarts_right.append(meanstart_right)


# adjust ticks

axes_2[0].set_xlim([-5,155])
axes_2[0].set_ylim(hist_y_lim)
axes_2[0].set_yticks([0,0.5,1])
axes_2[0].set_yticklabels([0,0.5,1])
axes_2[0].yaxis.set_minor_locator(MultipleLocator(0.25))
axes_2[0].yaxis.set_major_locator(MultipleLocator(0.5))
axes_2[0].xaxis.set_major_locator(MultipleLocator(100))
axes_2[0].xaxis.set_minor_locator(MultipleLocator(50))

axes_2[0].set_ylabel(hists_y_label,size=MEDIUM_SIZE)
axes_2[2].set_xlabel("$x$ (mm)",size=MEDIUM_SIZE)

set_grid(axes_2)


# plt.scatter(loading_contrast, x_plot[index_start])
# plt.xlabel("LC")
# plt.ylabel("nucleation point")
# plt.show()


#### drawing










temp_nice_plot_large(y_min,x_plot,ax=ax_drawing,ylim=None)

ax_drawing.set_xlim([0,150])
ax_drawing.set_ylim([-0.3,0.12])

ax_drawing.scatter(125, -0.05, s=20, marker='*', color='r', zorder=3)
ax_drawing.plot([75,75],[-0.2,0.12],linestyle='--',c='k')
ax_drawing.plot([125,125],[-0.2,0.12],linestyle='--',c='k')

ax_drawing.plot([75,125],[-0.2,-0.2],c='k',linewidth=.5)

ax_drawing.arrow(122,-0.2,1,0,length_includes_head = True, head_width=0.02, head_length = 3,edgecolor = None, color = "k", width = 0.001, shape = "full")
ax_drawing.arrow(78,-0.2,-1,0,length_includes_head = True, head_width=0.02, head_length = 3,edgecolor = None, color = "k", width = 0.001, shape = "full")

ax_drawing.text(100,-0.16,"$d_{nuc}$",size=SMALL_SIZE,horizontalalignment='center')
ax_drawing.axis("off")












# control smoothing of the data (rolling average) and starting point
roll_smooth=5
start=0
n_plot=10

## Location of the data inside the file
# channels containing the actual strain gages
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])
# channels containing the normal and tangetial force
forces_channels = [32,33]
# channel containing the trigger
trigger_channel = 34

## Parameters
# default x, just in case there's nothing in the saved params
x=np.array([1,2,3,4,5,8,11,12,13,14])*1e-2 -0.005
# Load the params file, and extract frequency of acquisition



## Plot data
## axes 3


#indexes_2[5]*=0
#indexes_2[6]+=0.00001
#indexes_2[7]+=0.000019

#indexes_2[4]*=0

# lines propag
xs=np.array([0.7,0.68,0.7])*1.005
ys=np.array([0.96,0.9,0.84])*0.995
line = mpl.lines.Line2D(xs, ys, lw=1, color='r', alpha=1)
fig.add_artist(line)

xs=[0.89,0.92]
ys=[0.53,0.4]
line = mpl.lines.Line2D(xs, ys, lw=1, color='r', alpha=1)
fig.add_artist(line)



color=[solid_in_granular_color]*5+[granular_color]+[solid_in_granular_color]*4

delta = -0.3

for i in range(10):
    axes_3[i].plot(1000*fast_time_2[start:]-1000*fast_time_2.mean()-delta,1000*(gages_2[3*(i+1)-1][start:]-gages_2[3*(i+1)-1][0:10000].mean()),color=color[i])
    #axes_3[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes_3[i].grid("both")
    # dashes
    axes_3[i].axvline(x=indexes_2[i]*1000-1000*fast_time_2.mean()-delta,c="k",linestyle="--")

axes_3[1].scatter(indexes_2[1]*1000-1000*fast_time_2.mean()-delta, 0, s=50, marker='*', color='r', zorder=3)


fig.text(0.7, 0.85, r"$v\sim 500$ m/s", size = INSET_SMALL_SIZE, rotation =0, c="r")
fig.text(0.8, 0.44, r"$v\sim 600$ m/s", size = INSET_SMALL_SIZE, rotation = 0,c="r")
#txt1.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))



axes_3[-1].set_xlabel('temps (ms)')
axes_3[4].set_ylabel(r'$\varepsilon_{xy}-\varepsilon_{xy}^0$ (mm/m)',size=MEDIUM_SIZE)


# adjust ticks
axes_3[-1].set_xlim([-0.24,0.24])
axes_3[-1].set_xticks([-0.2,-0.1,0,0.1,0.2])
axes_3[-1].set_xticklabels(["-$0.2$","-$0.1$","$0$","$0.1$","$0.2$"])
axes_3[-1].set_ylim([-0.15,0.06])
axes_3[-1].set_yticks([-0.1,0])
axes_3[-1].set_yticklabels(["-$0.1$","$0$"])



for i in range(9):
    plt.setp(axes_3[i].get_yticklabels(), visible=False)
    plt.setp(axes_3[i].get_xticklabels(), visible=False)




labels = [r"$x=5 \pm 0.25$ mm"]+[r"$x={}$".format(i) for i in [int(x_plot[i]) for i in range(1,len(x_plot))]]

# Iterate through each subplot and add data
for i, ax in enumerate(axes_3):
    ax.annotate(labels[i], xy=(0.025, 0.1), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)





mean_lc=data_loaded_2["mean_lc"]
manip_num=np.array(data_loaded_2["manip_num"])
LC_there = mean_lc[manip_num==chosen_manip_fig41_sec][0]


axes_3[0].annotate("${}={:.2f}$".format(LC_name_short,LC_there), xy=(0.75, 0.8), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)



set_grid(axes_3)























##


def my_mean(x):
    m = 0
    c = 0
    for xi in x:
        if xi!= 0:
            m+=xi
            c+=1
    if m==0:
        return(np.nan)
    return(m/c)

nuc_pt = x[index_start]
ell_nuc = 1000*(np.abs(nuc_pt-0.075))

ell_nuc_2 = nuc_pt*(nuc_pt>0.075)
ell_nuc_3 = nuc_pt*(nuc_pt<0.075)

xxs = np.arange(0,15,1)*10

lc_per_ell_nuc = [np.mean(loading_contrast[ell_nuc == xxs[i]]) if len(loading_contrast[ell_nuc == xxs[i]])!=0 else np.nan for i in range(15)]

#lc_list_nuc = np.linspace(-1,2,4)
#lc_list_nuc = np.linspace(-1,2,8)
#lc_list_nuc = np.linspace(-1,2,15)
lc_list_nuc = np.array([-1,-0.5,0,1,2])


ell_per_lc_nuc = np.array(
        [np.mean(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1] ,
                loading_contrast>lc_list_nuc[i])
                ]
            )/1000
        if len(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1],
                loading_contrast>lc_list_nuc[i] ,
                np.logical_not(from_solid)
                )
                ]
            )!=0
        else np.nan
        for i in range(len(lc_list_nuc)-1)]
        )


ell_per_lc_nuc_width = np.array(
        [np.std(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1] ,
                loading_contrast>lc_list_nuc[i] ,
                np.logical_not(from_solid)
                )
            ])/1000
        if len(
            ell_nuc[np.logical_and(
                loading_contrast<lc_list_nuc[i+1],
                loading_contrast>lc_list_nuc[i])
                ]
            )!=0
        else np.nan
        for i in range(len(lc_list_nuc)-1)]
        )







ell_per_lc_nuc_solid = np.mean(ell_nuc[from_solid])/1000


ell_per_lc_nuc_width_solid = np.std(ell_nuc[from_solid])/1000

lc_solid_mean = np.mean(loading_contrast[from_solid])










ax_4.scatter((lc_list_nuc[:-1]+lc_list_nuc[1:])/2,ell_per_lc_nuc*1000/15,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=3,
        edgecolors="k",linewidth=0.01)


ax_4.errorbar((lc_list_nuc[:-1]+lc_list_nuc[1:])/2,ell_per_lc_nuc*1000/15,yerr = ell_per_lc_nuc_width*1000/15,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)




ax_4.scatter(lc_solid_mean,ell_per_lc_nuc_solid*1000/15,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=3,
        edgecolors="k",linewidth=0.01)


ax_4.errorbar(lc_solid_mean,ell_per_lc_nuc_solid*1000/15,yerr = ell_per_lc_nuc_width_solid*1000/15,
               fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)










#ax_4.legend()
ax_4.set_xlim(-1.1,2)
ax_4.set_ylim(1.7,4.3)
ax_4.set_xlabel("${}$".format(LC_name_short),labelpad = 1.5)
ax_4.set_ylabel(r"$\frac{\left\langled_{nuc}\right\rangle}{\ell_{hole}\,/\,2}$")

ax_4.set_xticks([-1,0,1,2])
ax_4.set_xticklabels(["-1","0","1","2"])
ax_4.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.yaxis.set_major_locator(MultipleLocator(1))


set_grid(ax_4)



"""
##
ell = np.array(ell)

xxs = np.arange(0,15,1)*10

lc_per_ell_round = [np.mean(lc_event[ell == xxs[i]]) if len(lc_event[ell == xxs[i]])!=0 else np.nan for i in range(15)]

lc_list_round = np.linspace(-1,2,15)



ell_per_lc_round = np.array(
        [np.mean(
            ell[np.logical_and(lc_event<lc_list_round[i+1],lc_event>lc_list_round[i])])/1000
        if len(
            ell[np.logical_and(lc_event<lc_list_round[i+1],lc_event>lc_list_round[i])]
            )!=0
        else np.nan
        for i in range(len(lc_list_round)-1)]
        )

plt.scatter(lc_per_ell_round,xxs/1000,label=r"$\left\langle C_{\sigma}\right\rangle$ per bin of $\ell$")
plt.scatter(lc_list_round[:-1],ell_per_lc_round,label = r"$\left\langle\ell\right\rangle$ per bin of $C_{\sigma}$")
plt.legend()

plt.show()



"""


################### Plot adjustment


axes_1[-1].xaxis.labelpad=1
axes_2[2].xaxis.labelpad=1

axes_1[4].yaxis.set_label_coords(-0.058, 0)
axes_2[0].yaxis.set_label_coords(-0.24, 0.5)
ax_4.yaxis.set_label_coords(-0.09, 0.5)



axes_3[-1].xaxis.labelpad=1
#axes_4[2].xaxis.labelpad=1

axes_3[4].yaxis.set_label_coords(-0.058, 0)
#axes_4[0].yaxis.set_label_coords(-0.065*5, 0.5)


#real_tight_layout(fig)
y=(size_fig_1[1]-2.5*mm)/size_fig_1[1]
fig.text(0.005,0.98,"a.",size=LETTER_SIZE, weight='bold')
fig.text(.5,0.98,"b.",size=LETTER_SIZE, weight='bold')
fig.text(0.005,0.27,"c.",size=LETTER_SIZE, weight='bold')
fig.text(.68,0.27,"d.",size=LETTER_SIZE, weight='bold')

fig.savefig(main_folder + "Figures_chap_article/figure_4.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_article/figure_4.pdf")
fig.savefig(main_folder + "Figures_chap_article/figure_4.svg")
plt.close('all')


### Figure pour Elsa


fig,ax_4 = plt.subplots(1)
fig.set_size_inches((90*mm,70*mm))



ax_4.scatter((lc_list_nuc[:-1]+lc_list_nuc[1:])/2,ell_per_lc_nuc*1000/15,
        c=main_plot_color,s=scatter_size*2,
        marker="d",zorder=3,
        edgecolors="k",linewidth=0.01)


ax_4.errorbar((lc_list_nuc[:-1]+lc_list_nuc[1:])/2,ell_per_lc_nuc*1000/15,yerr = ell_per_lc_nuc_width*1000/15,
                 fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)




ax_4.scatter(lc_solid_mean,ell_per_lc_nuc_solid*1000/15,
        c=solid_solid_color,s=scatter_size*4/3,
        marker="o",zorder=3,
        edgecolors="k",linewidth=0.01)


ax_4.errorbar(lc_solid_mean,ell_per_lc_nuc_solid*1000/15,yerr = ell_per_lc_nuc_width_solid*1000/15,
               fmt=" ",capsize=error_bar_width,color="k",
               ecolor=error_bar_color,
               elinewidth=.5,alpha=error_bar_alpha,markeredgewidth=markeredgewidth)










#ax_4.legend()
ax_4.set_xlim(-1.1,2)
ax_4.set_ylim(1.7,4.3)
ax_4.set_xlabel("${}$".format(LC_name_short),labelpad = 1.5)
ax_4.set_ylabel(r"$\dfrac{\left\langled_{nuc}\right\rangle}{\ell_{hole}\,/\,2}$")

ax_4.set_xticks([-1,0,1,2])
ax_4.set_xticklabels(["-1","0","1","2"])
ax_4.yaxis.set_minor_locator(MultipleLocator(0.5))
ax_4.yaxis.set_major_locator(MultipleLocator(1))


set_grid(ax_4)

plt.tight_layout()


plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_4_d_only.png",dpi=dpi_global)
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_4_d_only.pdf")
plt.savefig(main_folder + "Figures_chap_article/figures_pour_elsa/figure_4_d_only.svg")
plt.close('all')











