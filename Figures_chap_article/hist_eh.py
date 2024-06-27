## Imports
# Science
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
# custom file


main_folder = 'E:/manuscrit_these/'
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


chosen_manip_fig41 = 37
chosen_event_fig41 = 3

reload = False
force_reload = False

test = np.load(main_folder + "data_for_figures/propag_eh.npy",allow_pickle=True).all()
locals().update(test)

loc_general="E:/2023-2024/2023-07-11-manips-10-voies/"




########################## create figure



# nested gridspecs, yeaaaaah...
fig = plt.figure()
fig.set_size_inches(size_fig_article_4_eh)





gs = gridspec.GridSpec(10,2,figure=fig, width_ratios=[2,1],
                      left=0.06, right=0.97,
                      top=0.98,bottom=0.1,
                      hspace=0, wspace=0.2)

axes_1=[]
for i in range(10):
    axes_1.append( fig.add_subplot(gs[i,0]) )
    axes_1[-1].sharex(axes_1[0])
    axes_1[-1].sharey(axes_1[0])

ax2=fig.add_subplot(gs[2:-2, 1])


########################## Data for propagation


# main folder
loc_folder=loc_general+"/manip_{}/".format(chosen_manip_fig41)
# name of the file / event
loc_file="event-00{}.npy".format(chosen_event_fig41)
# name of the reference file, containing unloaded signal, usually event-001
loc_file_zero = "event-001.npy"
# parameters file
loc_params="parameters.txt"

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







if force_reload or reload :
    print("reloading data for Figure 4")

    exec(load_params(loc_folder+loc_params))
    sampling_freq_in = clock/10
    # Create the location strings
    loc=loc_folder+loc_file
    loc_zero=loc_folder+loc_file_zero
    # Number of channels
    nchannels = len(gages_channels)

    # Fast acquicition
    # Load data


    data=np.load(loc,allow_pickle=True)
    data_zero=np.load(loc_zero,allow_pickle=True)

    # smooth data
    data=smooth(data,roll_smooth)
    data=np.transpose(np.transpose(data)-np.mean(data_zero,axis=1))

    # assign specific channels to specific variables
    forces=data[forces_channels,:]
    mu = data[forces_channels[1],:].mean() / data[forces_channels[0],:].mean()
    gages = data[gages_channels]
    gages_zero = data_zero[gages_channels]
    gages=np.transpose(np.transpose(gages)-np.mean(gages_zero,axis=-1))
    fast_time=np.arange(len(gages[0]))/sampling_freq_in

    for i in range(nchannels//3):
        ch_1=gages[3*i]
        ch_2=gages[3*i+1]
        ch_3=gages[3*i+2]
        ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
        ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
        #ch_1,ch_2,ch_3=eps_to_sigma(ch_1,ch_2,ch_3,E=E,nu=nu)
        gages[3*i]=ch_1
        gages[3*i+1]=ch_2
        gages[3*i+2]=ch_3


    # axes 1

    indexes=np.load(loc+"_times_hand_picked.npy")



## plot data
# lines propag
xs=[0.248,0.315]
ys=[0.41,0.15]
line = mpl.lines.Line2D(xs, ys, lw=1, color='r', alpha=1)
fig.add_artist(line)


xs=[0.43,0.485]
ys=[0.58,0.95]
line = mpl.lines.Line2D(xs, ys, lw=1, color='r', alpha=1)
fig.add_artist(line)



indexes_bis = indexes + 1e-3*np.array([ 0.015,
                    0.037,
                    0.03,
                    0.023,
                    0.023,
                    0.01,
                    0.01,
                    0.004,
                    0.002,
                    0.01
        ]
    )


color=[solid_in_granular_color]*5+[solid_solid_color]+[solid_in_granular_color]*4

for i in range(10):
    axes_1[i].plot(1000*fast_time[start:]-1000*fast_time.mean(),1000*(gages[3*(i+1)-1][start:]-gages[3*(i+1)-1][0:10000].mean()),color=color[i])
    axes_1[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes_1[i].grid("both")
    # dashes
    if i!=5:
        axes_1[i].axvline(x=indexes_bis[i]*1000-1000*fast_time.mean(),c="k",linestyle="--")

axes_1[6].scatter(indexes_bis[6]*1000-1000*fast_time.mean(), 0, s=50, marker='*', color='r', zorder=3)
fig.text(0.34, 0.82, r"$v\sim 1\,600$ m/s", size = INSET_SMALL_SIZE, rotation =0, c="r")
fig.text(0.32, 0.25, r"$v\sim 1\,000$ m/s", size = INSET_SMALL_SIZE, rotation = 0,c="r")
#txt1.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))



axes_1[-1].set_xlabel('temps (ms)')
axes_1[4].set_ylabel(r'$\varepsilon_{xy}-\varepsilon_{xy}^0$ (mm/m)',size=MEDIUM_SIZE)


# adjust ticks
axes_1[-1].set_xlim([-0.12,0.12])
axes_1[-1].set_xticks([-0.1,-0.05,0,0.05,0.1])
axes_1[-1].set_xticklabels(["-$0.1$","-$0.05$","$0$","$0.05$","$0.1$"])
axes_1[-1].set_ylim([-0.2,0.1])
axes_1[-1].set_yticks([-0.2,0])
axes_1[-1].set_yticklabels(["-$0.2$","$0$"])



for i in range(9):
    plt.setp(axes_1[i].get_yticklabels(), visible=False)
    plt.setp(axes_1[i].get_xticklabels(), visible=False)




labels = [r"$x=5 \pm 0.25$ mm"]+[r"$x={}$".format(i) for i in [int(x_plot[i]) for i in range(1,len(x_plot))]]

# Iterate through each subplot and add data
for i, ax in enumerate(axes_1):
    ax.annotate(labels[i], xy=(0.025, 0.15), xycoords='axes fraction',
                xytext=(10, 0), textcoords='offset points',
                va="center",size=INSET_SMALL_SIZE)





# LC :
loc_folder=loc_general

if reload or force_reload:
    data_loaded=np.load(loc_folder+"python_plots/summary_data.npy",allow_pickle=True).all()


mean_lc=data_loaded["mean_lc"]
manip_num=np.array(data_loaded["manip_num"])
LC_here = mean_lc[manip_num==chosen_manip_fig41][0]




set_grid(axes_1)



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
    arc_radius = 0.03*dilat
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
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.07*dilat), 2*width_patch, 0.03*dilat, color=solid_in_granular_color,alpha=1)
        else:
            square_patch = patches.Rectangle((xi-width_patch, y_min+0.07*dilat), 2*width_patch, 0.03*dilat, color=granular_color,alpha=1)
        ax.add_patch(square_patch)



## Data location
# location of the main folder containing all the "manip_..." subfolders

loc_folder=loc_general
loc_figures = loc_folder + "histograms/"
loc_manip = "manip_{}/"
loc_params="parameters.txt"



## Load all data


if reload or force_reload:
    data_loaded_3=np.load(loc_folder+"python_plots/summary_data_2.npy",
                    allow_pickle=True).all()



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



ax2.bar(x_plot, histogram_start_solid/histogram_start_solid.sum(),
        width=width,
        color = solid_solid_color,
        linewidth=.1,edgecolor='k')
temp_nice_plot(y_min,x_plot,ax=ax2,ylim=None)
ax2.grid(True,which="both")
meanstart = (x_plot*histogram_start_solid).sum()/histogram_start_solid.sum()
meanstarts_left.append(meanstart)
meanstarts_right.append(meanstart)

# adjust ticks

ax2.set_xlim([-5,155])
ax2.set_ylim(hist_y_lim)
ax2.set_yticks([0,0.5,1])
ax2.set_yticklabels([0,0.5,1])
ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
ax2.yaxis.set_major_locator(MultipleLocator(0.5))
ax2.xaxis.set_major_locator(MultipleLocator(100))
ax2.xaxis.set_minor_locator(MultipleLocator(50))

ax2.set_ylabel(hists_y_label,size=MEDIUM_SIZE)
ax2.set_xlabel("$x$ (mm)",size=MEDIUM_SIZE)

set_grid(ax2)





fig.text(0.005,0.965,"a.",size=LETTER_SIZE, weight='bold')
fig.text(.63,0.965,"b.",size=LETTER_SIZE, weight='bold')

fig.savefig(main_folder + "Figures_chap_article/hist_eh.png",dpi=dpi_global)
fig.savefig(main_folder + "Figures_chap_article/hist_eh.pdf")
fig.savefig(main_folder + "Figures_chap_article/hist_eh.svg")
plt.close('all')





#
if reload or force_reload:

    to_save={}


    to_save["labels"]=labels
    to_save["x_plot"]=x_plot
    to_save["histogram_start_solid"]=histogram_start_solid
    to_save["histograms_start"]=histograms_start
    to_save["chosen_manip_fig41"]=chosen_manip_fig41
    to_save["chosen_event_fig41"]=chosen_event_fig41


    #to_save["lc_per_ell_round"]=lc_per_ell_round
    #to_save["possible_ells"]=xxs
    #to_save["lc_list_round"]=lc_list_round[-1]
    #to_save["ell_per_lc_round"]=ell_per_lc_round

    to_save["fast_time"]=fast_time
    to_save["gages"]=gages
    to_save["indexes"]=indexes


    to_save["data_loaded"]=data_loaded
    to_save["data_loaded_3"]=data_loaded_3


    np.save(main_folder + "data_for_figures/propag_eh.npy",to_save)







