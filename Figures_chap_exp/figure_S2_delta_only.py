###
main_folder = 'E:/manuscrit_these/'

main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())




loaded_data = np.load(main_folder + "data_for_figures/figure_S2.npy",allow_pickle= True).all()
locals().update(loaded_data)

###



def sliding_plot(x,disp,ax,vlines = None,labels=True,suptitle=False,color="blue",alpha=1):

    if not vlines is None:
        for v in vlines:
            #ax.axvline(v,color="black",linestyle=dash_line_style ,linewidth=0.4,alpha= secondary_plot_alpha)
            ax.plot([v,v],[0,disp[np.abs(x - v).argmin()]],
                    color=dashed_color,linestyle=dash_line_style,
                    linewidth=0.1,alpha= 0.2)

    ax.grid(True,which="both")

    ax.plot(x,disp,color=color,alpha=alpha)


    if labels:
        ax.set_ylabel(tot_sliding_short )

    if suptitle:
        ax.set_title(suptitle)


def creep_plot(x,creep,ax,labels=True,suptitle=False,color="blue",alpha=1,marker=scatter_marker,s=scatter_size,vlines=False ):
    if vlines:
        for i in range(len(x)):
            ax.plot([x[i],x[i]],[creep[i],100],
                    color=dashed_color,linestyle=dash_line_style,
                    linewidth=0.1,alpha=0.2)

    ax.plot(x,creep,color=color,alpha=alpha)
    ax.scatter(x,creep,marker=marker ,s=s ,c=color,alpha = 1 ,zorder=10,edgecolors="k",linewidth=0.01)

    ax.grid(True,which="both")

    if labels:
        ax.set_ylabel(sliding_perc_name_short )

    if suptitle:
        ax.set_title(suptitle)




def mu_plot(t,mu,ax,timings=None,linecolor=main_plot_color):
    a=next((i for i, b in enumerate(mu[7000:]>0.05) if b), None)+7000
    start = max(a-5200,0)
    ax.plot(t[start:]-t[start],mu[start:],color=linecolor )


def ramp(l,v,f=100):
    return(v*np.arange(0,l,1)/f)



##
def findevents(signal, sensit):
    typical_wait_time = 60
    # signal = np.convolve(signal, np.ones(5)/5, mode='valid')  # Smoothing signal, if needed
    diff = np.abs(signal[2:] - signal[:-2])
    m = np.mean(diff)
    diff -= m
    index = np.where(diff > sensit)[0]
    index = np.append(index, index[-1] + 2 * typical_wait_time)
    diffindex = np.abs(-index[1:] + index[:-1])
    found_events = [index[0]]
    found_events.extend(index[np.where(diffindex > typical_wait_time)[0] + 1])
    found_events.pop()  # Remove the last element
    return np.array(found_events)

# Assuming the variables such as avgtop, avgbot, xyresolution, tInit, tEnd, detection_grain,
# listAllCircles, sensit, space_after_event_temp, space_before_event, removelast,
# avgtopleft, avgbotleft, avgtopright, avgbotright, avgtopcenter, avgbotcenter, realtime
# are already defined and initialized in your Python environment



totslip = -data_cam_cut
realtime= np.arange(0,len(data_cam_cut))/100-0.21-5

found_events = findevents(data_cam_cut, 0.01)


if len(found_events) > 1:
    space_after_event_temp = min(np.fix(np.diff(found_events) / 10))
else:
    space_after_event_temp = 50

space_after_event = max(min(50, space_after_event_temp), 15)
space_before_event = 5


##

delta = np.sum(totslip[found_events + space_after_event] - totslip[found_events - space_before_event])

delta_tot = totslip[-1] - totslip[0]

creeped_distance = delta_tot - delta


###
# Compute creep proportion to seek convergence
creep = totslip

delta = []
delta_tot = []
totslip_partial = []

for i in range(len(found_events)):
    delta.append(np.sum(creep[found_events[0:i+1] + space_after_event] - creep[found_events[0:i+1] - space_before_event]))
    delta_tot.append(creep[found_events[i] + space_after_event] - creep[0])

    totslip_partial.append(abs(np.mean(creep[found_events[i] + int(space_after_event/2):found_events[i] + space_after_event]) ) )

creeped_distance_frac = (np.array(delta_tot) - np.array(delta)) / np.array(totslip_partial)
creeped_distance = np.array(delta_tot) - np.array(delta)






## FIGURE

time_lim_1=[-1,51]
time_lim_2=[-0.5,55]



fig = plt.figure(layout=None)
fig.set_size_inches(size_fig_S2_res_only)

nrows=12


gs = fig.add_gridspec(nrows=3, ncols=2,
                      hspace=0, height_ratios=[0.5,0.1,0.5],
                      left=0.08)


ax_left=fig.add_subplot(gs[:,0])
ax_br = [fig.add_subplot(gs[0,1], sharex = ax_left),fig.add_subplot(gs[2,1], sharex = ax_left)]

##


ax_left.plot(time_cam_script,-data_cam_script,label="suivi sous-pixel",c=u"#2ca02c")

ax_left.xaxis.set_minor_locator(MultipleLocator(10))
ax_left.xaxis.set_major_locator(MultipleLocator(20))
ax_left.yaxis.set_minor_locator(MultipleLocator(0.1))
ax_left.yaxis.set_major_locator(MultipleLocator(0.2))

ax_left.set_ylim([0,1.30])
ax_left.set_xlim([0,58])
set_grid(ax_left)
ax_left.set_ylabel('$\delta_{tot}$ (mm)')


for i in range(len(found_events)):
    ax_left.axvline(time_simul_script[found_events[i]], linestyle='--', color='k',linewidth=.5)


"""
coords = [0.3,   0.16,    0.15,  0.3] #left, bottom, width, height
axin01=fig.add_axes(coords)



axin01.plot(time_simul_script,-data_simul_script,label="simul",zorder=3)
axin01.plot(time_prof_script,-data_prof_script,label="profilo")
axin01.plot(time_cam_script,-data_cam_script,label="camera")
#axin01.plot(time_cam_script,-data_cam_script_pix,label="camera pix")

axin01.plot(time_simul_script,-data_simul_script+0.008,linewidth=.5,c="k",linestyle=':')
axin01.plot(time_simul_script,-data_simul_script-0.008,linewidth=.5,c="k",linestyle=':')

axin01.set_xlim([14,26])
axin01.set_ylim([0.300,0.60])
axin01.xaxis.set_minor_locator(MultipleLocator(1))
axin01.xaxis.set_major_locator(MultipleLocator(2))
axin01.yaxis.set_minor_locator(MultipleLocator(0.050))
axin01.yaxis.set_major_locator(MultipleLocator(0.10))


axin01.grid(False)
axin01.set_facecolor("w")
set_up_inset(axin01)

"""



# bottom right

ax_br[0].plot(realtime[found_events], creeped_distance, '-', color='red', label='Left',marker="x")
#ax_br[0].plot([0,60], [0,0.6], ':', color='k')
ax_br[0].set_ylabel('$\delta_{IE}$ (mm)')
ax_br[0].set_ylim([0,0.65])
ax_br[0].yaxis.set_minor_locator(MultipleLocator(0.1))
ax_br[0].yaxis.set_major_locator(MultipleLocator(0.2))
plt.setp(ax_br[0].get_xticklabels(), visible=False)

for i in range(len(found_events)):
    ax_br[0].axvline(time_simul_script[found_events[i]], linestyle='--', color='k',linewidth=.5)
    ax_br[1].axvline(time_simul_script[found_events[i]], linestyle='--', color='k',linewidth=.5)



ax_br[1].plot(realtime[found_events], creeped_distance_frac, '-', color='red', label='Left',marker="x")
ax_br[1].set_ylabel('$S=\delta_{IE}/\delta_{tot}$')
ax_br[1].set_ylim([0.4,0.6])
#ax_br[1].axhline(50,c="k",linestyle=":")
ax_br[1].yaxis.set_minor_locator(MultipleLocator(0.05))
ax_br[1].yaxis.set_major_locator(MultipleLocator(0.1))


ax_left.set_xlabel('temps (s)')

ax_left.get_yaxis().labelpad=6
set_grid(ax_br)



fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.13, wspace = 0.25)


y=0.95
fig.text(0.01,y,"a",size=LETTER_SIZE, weight="bold")
fig.text(.50,y,"b",size=LETTER_SIZE, weight="bold")
fig.text(.50,y/1.9,"c",size=LETTER_SIZE, weight="bold")


plt.savefig(main_folder + "Figures_chap_exp/figure_S2_delta_only.png",dpi=1200)
plt.savefig(main_folder + "Figures_chap_exp/figure_S2_delta_only.pdf")
plt.savefig(main_folder + "Figures_chap_exp/figure_S2_delta_only.svg")

#plt.show()
plt.close('all')






