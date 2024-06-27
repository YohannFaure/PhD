import cv2
import numpy as np
from matplotlib import pyplot as plt

main_folder = 'E:/manuscrit_these/'
main_folder = '../'

with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


# Load the first image
#larger_image = cv2.imread('E:/2023-2024/2024-02-15-extract_old_images_for_tracking/cine2_1.png', 0)
larger_image = cv2.imread(main_folder + 'data_for_figures/cine2_1.png', 0)

# Load template
#template = cv2.imread('E:/2023-2024/2024-02-15-extract_old_images_for_tracking/template.png', 0)
template = cv2.imread(main_folder + 'data_for_figures/template.png', 0)

# Perform template matching
result = cv2.matchTemplate(larger_image, template, cv2.TM_CCOEFF_NORMED)

# Get the size of the template
w, h = template.shape[::-1]

# Find the maximum correlation value and its position
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
#cv2.rectangle(larger_image, top_left, bottom_right, 255, 2)




# Load images
image1 = result
image2 = larger_image
image5 = template

# Create subplots
fig, axs = plt.subplots(1, 6, figsize=(10, 2),gridspec_kw={'width_ratios': [0.02,0.83,0.06, 2.7, -0.02, 2]})

# Plot large images in a square disposition
axs[1].imshow(image5,cmap='gray')
expim1 = np.exp(image1)

axs[1].annotate(r"⌀",(2.5,15),size=10,family='Cambria',color="k",annotation_clip=False)
axs[1].annotate(" = 1.3 mm",(3.5,15),size=INSET_SMALL_SIZE,color="k",annotation_clip=False)

pos = axs[3].imshow(expim1,cmap = 'winter')#cmap='gray')
axs[5].imshow(image2,cmap='gray')

cbar = fig.colorbar(pos, ax=axs[3], location='right', anchor=(-0.2, 0.5), shrink=0.9)

cbar.set_label('corrélation\n(u.a.)', rotation=270,size = 10, labelpad = 14)

cbar.set_ticks([np.min(expim1),np.max(expim1)],labels=[0,1])

xx=top_left[0]
yy=top_left[1]
ww=bottom_right[0]-xx
hh=bottom_right[1]-yy


rect2 = mpl.patches.Rectangle( [xx,yy], ww, hh, linewidth=2, edgecolor='red', facecolor='none',zorder=10)

axs[5].add_patch(rect2)


# Hide axes
for ax in axs.flatten():
    ax.axis('off')


plt.tight_layout()

fig.set_size_inches(size_correl_1)

fig.text(0.005,0.88,"a.",fontweight="bold",size=12)
fig.text(0.175,0.88,"b.",fontweight="bold",size=12)
fig.text(0.615,0.88,"c.",fontweight="bold",size=12)



plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_1.png",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_1.svg",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_1.pdf",dpi=600)
plt.close()


##

fig,axes=plt.subplots(1,2,gridspec_kw={'width_ratios': [2,1]})
fig.set_size_inches(size_correl_2)
image1_norm = (image1-np.min(image1))
image1_norm = image1_norm / np.max(image1_norm)

x=range(157)
y=range(59)
axes[0].plot(x,image1_norm[25],c=main_plot_color)
axes[1].plot(image1_norm[:,69],y,c=main_plot_color)
axes[0].set_xlabel("x (px)")
axes[1].set_xlabel("correl. (U.A.)")
axes[0].set_ylabel("correl. (U.A.)")
axes[1].set_ylabel("y (px)")
axes[0].axvline(max_loc[0],c="k",linestyle=":")
axes[1].axhline(max_loc[1],c="k",linestyle=":")
axes[0].grid()
axes[1].grid()

axes[0].set_yticks([0,0.5,1])
axes[0].set_yticklabels(["0","0.5","1"])
axes[0].set_ylim([0,1])

axes[0].set_xticks([0,50,100,150])
axes[0].set_xticklabels(["0","50","100","150"])
axes[0].set_xlim([0,156])

axes[1].set_xticks([0,0.5,1])
axes[1].set_xticklabels(["0","0.5","1"])
axes[1].set_xlim([0,1])

axes[1].set_yticks([0,25,50])
axes[1].set_yticklabels(["0","25","50"])
axes[1].set_ylim([0,58])



axes[0].xaxis.set_minor_locator(MultipleLocator(25))
axes[1].yaxis.set_minor_locator(MultipleLocator(25))

axes[0].yaxis.set_minor_locator(MultipleLocator(0.25))
axes[1].xaxis.set_minor_locator(MultipleLocator(0.25))

set_grid(axes)

plt.tight_layout()
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_2.png",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_2.svg",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_2.pdf",dpi=600)

plt.close()



##
"""

n_cor = 10

x_i,y_i=25,69

small_cor = np.copy(image1[x_i-n_cor:x_i+n_cor+1, y_i-n_cor:y_i+n_cor+1])
small_cor -= np.min(small_cor)
small_cor /= np.max(small_cor)

small_cor[small_cor<0.2]=0




def gaussian_interp(C, xPix, yPix):
    hC, wC = C.shape

    imax = np.argmax(np.abs(C))
    ylocmaxC, xlocmaxC = np.unravel_index(imax, C.shape)

    Cint = (C * 254 + 1)
    xlocmaxC = max(1, min(wC - 1, xlocmaxC))
    ylocmaxC = max(1, min(hC - 1, ylocmaxC))

    Zxm1 = Cint[ylocmaxC, xlocmaxC - 1]
    Zx = Cint[ylocmaxC, xlocmaxC]
    Zxp1 = Cint[ylocmaxC, xlocmaxC + 1]

    Zym1 = Cint[ylocmaxC - 1, xlocmaxC]
    Zy = Cint[ylocmaxC, xlocmaxC]
    Zyp1 = Cint[ylocmaxC + 1, xlocmaxC]

    Dx = (np.log(Zxm1) - np.log(Zxp1)) / (2 * (np.log(Zxp1) - 2 * np.log(Zx) + np.log(Zxm1)))
    Dy = (np.log(Zym1) - np.log(Zyp1)) / (2 * (np.log(Zyp1) - 2 * np.log(Zy) + np.log(Zym1)))


    xSub = xlocmaxC + Dx
    ySub = ylocmaxC + Dy

    return xSub, ySub



x,y = gaussian_interp(small_cor,5,5)
#plt.imshow(small_cor)
#plt.scatter(x,y)
#plt.show()
print(x,y)




from scipy.optimize import curve_fit

def gaussian_2d(xy_mesh, xo, yo, sigma):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = 1 / (2 * sigma**2)
    c = 1 / (2 * sigma**2)
    g =  np.exp(-(a * ((x - xo)**2) + c * ((y - yo)**2)))
    return g.ravel()

n_cor = 10


small_cor = np.copy(image1[x_i-n_cor:x_i+n_cor+1, y_i-n_cor:y_i+n_cor+1])
small_cor -= np.min(small_cor)
small_cor /= np.max(small_cor)

small_cor[small_cor<0.5]=0


def fit_2d_gaussian(data):

    hC, wC = data.shape

    imax = np.argmax(np.abs(data))
    ylocmaxC, xlocmaxC = np.unravel_index(imax, data.shape)

    Cint = data
    xlocmaxC = max(1, min(wC - 1, xlocmaxC))
    ylocmaxC = max(1, min(hC - 1, ylocmaxC))

    Zxm1 = Cint[ylocmaxC, xlocmaxC - 1]
    Zx = Cint[ylocmaxC, xlocmaxC]
    Zxp1 = Cint[ylocmaxC, xlocmaxC + 1]

    Zym1 = Cint[ylocmaxC - 1, xlocmaxC]
    Zy = Cint[ylocmaxC, xlocmaxC]
    Zyp1 = Cint[ylocmaxC + 1, xlocmaxC]

    newdata = np.array([ [0,0,Zym1,0,0],[0,Zxm1,Zx,Zxp1,0],[0,0,Zyp1,0,0] ])
    x = np.linspace(0, newdata.shape[1] - 1, newdata.shape[1])
    y = np.linspace(0, newdata.shape[0] - 1, newdata.shape[0])
    x, y = np.meshgrid(x, y)

    initial_guess = (newdata.shape[1]/2, newdata.shape[0]/2, 2)
    popt, pcov = curve_fit(gaussian_2d, (x, y), newdata.ravel(), p0=initial_guess)

    Dx,Dy = popt[0],popt[1]

    xSub = xlocmaxC + Dx -2
    ySub = ylocmaxC + Dy -1

    return xSub, ySub , popt[2]

xx,yy,popt = fit_2d_gaussian(small_cor)
#plt.imshow(small_cor)
#plt.scatter(xx,yy,c="r")
#plt.show()

print(xx,yy)


space = np.linspace(0,20,21)

fitted = gaussian_2d(np.meshgrid(space,space), xx, yy, 2).reshape(21,21)

fig, axes = plt.subplots(1,3)
axes[0].imshow(small_cor,vmin = 0, vmax = 1)
axes[0].scatter(xx,yy,c="r")
axes[1].imshow(fitted,vmin = 0, vmax = 1)
axes[2].imshow(small_cor-fitted,vmin = 0, vmax = 1)
plt.show()

#


fig,axes=plt.subplots(1,2,gridspec_kw={'width_ratios': [1,1]})
axes[0].plot(small_cor[10,:])
axes[1].plot(small_cor[10,:])
axes[0].plot(fitted[10,:])
axes[1].plot(fitted[10,:])
axes[0].set_xlabel("x (px)")
axes[1].set_xlabel("correl. (U.A.)")
axes[0].set_ylabel("correl. (U.A.)")
axes[1].set_ylabel("y (px)")
axes[0].grid()
axes[1].grid()
plt.tight_layout()
plt.show()
##

"""

###



import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

radius = 7

# Load the image
img = cv2.imread(main_folder + 'data_for_figures/cine2_1.png', 0)


# Apply GaussianBlur to reduce noise and improve circle detection
img_blur = cv2.blur(img, (3, 3))

# Apply Hough Circle Transform
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                            param1=50, param2=11, minRadius=5, maxRadius=9)


if circles is not None:
    # Convert the circle parameters to integers
    circles = np.uint16(np.around(circles))

    # Draw detected circles
    circs=[]
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        # Draw the outer circle
        circs.append(mpl.patches.Circle((center), radius,facecolor='none',edgecolor='r',linewidth=1) )


    # Display the image with detected circles

    fig,ax=plt.subplots(1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for c in circs:
        ax.add_patch(c)
    #plt.title('Detected Circles')
    plt.axis('off')


plt.tight_layout()
fig.set_size_inches(size_correl_0)

"""
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_0.png",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_0.svg",dpi=600)
plt.savefig(main_folder + "Figures_chap_exp/correlation_grains_0.pdf",dpi=600)
"""

plt.close()






