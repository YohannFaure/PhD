import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



main_folder = 'E:/manuscrit_these/'

main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())

#location = "E:/2023-2024/2024-02-09-video-comparaison-cam/tiff/"
location = main_folder + "data_for_figures/tiff/"

files = ["2511_100fps_dataraw.mat",
         "2511_max_speed_dataraw.mat",
         "2511_max_speed_low_light_dataraw.mat",
         "6410_100fps_dataraw.mat",
         "6410_max_speed_dataraw.mat",
         "6410_comme_2511_iso40000_dataraw.mat",
         "6410_comme_2511_iso80000_dataraw.mat"]

resolution=8 # px/mm

# manip
i=0
# fps
fps=100
# grain
j=7
#
start=10000

n_roll = 40

## import data
file = files[i]
data = scio.loadmat(location+file)
centersXs=data['centersXs']
centersYs=data['centersYs']
xSubs=data['xSubs']
ySubs=data['ySubs']

centersXs=np.nan_to_num(centersXs)
centersYs=np.nan_to_num(centersYs)
xSubs=np.nan_to_num(xSubs)
ySubs=np.nan_to_num(ySubs)


## Plot


initval=np.arange(0,1,1/xSubs.shape[0])*2000
sub=xSubs.transpose()-xSubs[:,start]+initval
pix=centersXs.transpose()-centersXs[:,start]+initval
#plt.plot(sub[10000:,j],alpha=.5)
#plt.plot(pix[10000:,j],alpha=.5,c="k")
#plt.show()



##

def rolling_std(arr, window):
    """
    Compute the rolling standard deviation of an array.

    Parameters:
    arr (numpy.ndarray): Input array.
    window (int): Size of the rolling window.

    Returns:
    numpy.ndarray: Array of rolling standard deviations.
    """
    # Calculate rolling sum
    rolling_sum = np.convolve(arr, np.ones(window), mode='valid')
    # Calculate rolling sum of squares
    rolling_sum_sq = np.convolve(arr**2, np.ones(window), mode='valid')
    # Calculate the mean
    mean = rolling_sum / window
    # Calculate the rolling variance
    variance = rolling_sum_sq / window - mean**2
    variance[variance<0]=0 # floating point errors
    # Take square root to get standard deviation
    rolling_std = np.sqrt(variance)
    return(rolling_std)


##

time=np.arange(0,len(sub)/fps,1/fps)

# import data
file = files[i]

data = scio.loadmat(location+file)
centersXs=data['centersXs']
centersYs=data['centersYs']
xSubs=data['xSubs']
ySubs=data['ySubs']

centersXs=np.nan_to_num(centersXs)
centersYs=np.nan_to_num(centersYs)
xSubs=np.nan_to_num(xSubs)
ySubs=np.nan_to_num(ySubs)

dist = np.sqrt(xSubs**2+ySubs**2)

# compute sigma
sigmarray_x=np.zeros((xSubs.shape[0],xSubs.shape[1]-n_roll+1))
sigmarray_y=np.zeros((ySubs.shape[0],ySubs.shape[1]-n_roll+1))
sigmarray_d=np.zeros((dist.shape[0],dist.shape[1]-n_roll+1))
for k in range(xSubs.shape[0]):
    sigmarray_x[k]=rolling_std(xSubs[k],n_roll)
    sigmarray_y[k]=rolling_std(ySubs[k],n_roll)
    sigmarray_d[k]=rolling_std(dist[k],n_roll)

sigmarray_x=np.median(sigmarray_x,axis=1)
sigmarray_y=np.median(sigmarray_y,axis=1)
sigmarray_d=np.median(sigmarray_d,axis=1)

sigma_x=np.median(sigmarray_x)
sigma_y=np.median(sigmarray_y)
sigma_d=np.median(sigmarray_d)

sigma = np.median(sigmarray_d)


reference = np.convolve(xSubs[j], np.ones(n_roll), mode='valid')/n_roll
reference=reference-np.mean(reference)






fig,axes=plt.subplots(1,2)

fig.set_size_inches(size_tracking)

axes[0].plot(time[start:]-100,1000*(centersXs[j]-np.mean(centersXs[j])+initval[i])[10000:]/resolution,c='y',alpha=0.5,label="suivi pixel")

axes[0].plot(time[start:]-100,1000*(xSubs[j]-np.mean(xSubs[j])+initval[i])[start:]/resolution,
         c="k",alpha=.5,linewidth=.5,label="sous-pixel")



rect = mpl.patches.Rectangle((40, 165), 10, 20, linewidth=1, edgecolor='red', facecolor='none',zorder=10)
axes[0].add_patch(rect)

axes[0].grid(which="both")

axes[0].set_ylabel("x (µm)")
axes[0].set_xlabel("temps (s)")


axes[1].plot(time[start:]-100,1000*(xSubs[j]-np.mean(xSubs[j])+initval[i])[10000:]/resolution,
         c='k',alpha=0.2,linewidth=.5)

axes[1].plot(time[start:-n_roll+1]-100,1000*(initval[i]+ sigma+reference)[start:]/resolution,
         c='g',alpha=1,linewidth=1)

axes[1].plot(time[start:-n_roll+1]-100,1000*(initval[i]- sigma+reference)[start:]/resolution,
         c='g',alpha=1,linewidth=1)

axes[1].plot(time[start:-n_roll+1]-100,1000*(initval[i]+reference)[start:]/resolution,
         c='b',alpha=1,linewidth=1)

axes[1].grid(which='Both')
axes[1].set_xlim([39,51])
axes[1].set_ylim([163,187])
rect2 = mpl.patches.Rectangle((40, 165), 10, 20, linewidth=1, edgecolor='red', facecolor='none',zorder=10)

axes[1].add_patch(rect2)
axes[1].set_xlabel("temps (s)")



axes[0].legend(facecolor = (1,1,1,1),framealpha=1)

axes[0].xaxis.set_major_locator(MultipleLocator(50))
axes[0].xaxis.set_minor_locator(MultipleLocator(25))
axes[0].yaxis.set_major_locator(MultipleLocator(100))
axes[0].yaxis.set_minor_locator(MultipleLocator(50))

axes[1].xaxis.set_major_locator(MultipleLocator(5))
axes[1].xaxis.set_minor_locator(MultipleLocator(2.5))
axes[1].yaxis.set_major_locator(MultipleLocator(10))
axes[1].yaxis.set_minor_locator(MultipleLocator(5))

set_grid(axes)

plt.tight_layout()

fig.text(0.01,0.95,"a.",size=12, weight="bold")
fig.text(0.52,0.95,"b.",size=12, weight="bold")


plt.savefig(main_folder + "Figures_chap_exp/tracking_grains.png",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/tracking_grains.svg")
plt.savefig(main_folder + "Figures_chap_exp/tracking_grains.pdf")
plt.close()