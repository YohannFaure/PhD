
main_folder = '../'


with open(main_folder + "parameters.py", 'r') as f:
        exec(f.read())


import cv2



# load images


image1 = cv2.imread(main_folder + "data_for_figures/imageseq/cine4.png")
image2 = cv2.imread(main_folder + "data_for_figures/imageseq/cine4_2.png")

# compute difference
difference = cv2.subtract(image1, image2)

# color the mask red
Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
difference[difference>70] = 255

# add the red mask to the images to make the differences obvious
image1[mask != 255] = [0, 0, 255]
image2[mask != 255] = [0, 0, 255]


image3 = cv2.imread(main_folder + "data_for_figures/imageseq/cine3.png")
image4 = cv2.imread(main_folder + "data_for_figures/imageseq/cine3_2.png")

# compute difference
difference_2 = cv2.subtract(image3, image4)

# color the mask red
Conv_hsv_Gray = cv2.cvtColor(difference_2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
difference_2[difference_2>70] = 255

# add the red mask to the images to make the differences obvious
image3[mask != 255] = [0, 0, 255]
image4[mask != 255] = [0, 0, 255]




fig,axes = plt.subplots(7,1)

fig.subplots_adjust(left=0,
                    bottom=0,
                    right=1,
                    top=1,
                    wspace=0,
                    hspace=0)


axes[0].imshow(image1)

axes[1].imshow(image2)

axes[2].imshow(difference)

axes[4].imshow(image3)

axes[5].imshow(image4)

axes[6].imshow(difference_2)

for i in range(7):
    axes[i].axis('off')



plt.show()
# store images



cv2.imwrite(main_folder + 'Figures_chap_conclusion/local_1.png', image1)
cv2.imwrite(main_folder + 'Figures_chap_conclusion/local_2.png', image2)
cv2.imwrite(main_folder + 'Figures_chap_conclusion/local_diff.png', difference)

cv2.imwrite(main_folder + 'Figures_chap_conclusion/total_1.png', image3)
cv2.imwrite(main_folder + 'Figures_chap_conclusion/total_2.png', image4)
cv2.imwrite(main_folder + 'Figures_chap_conclusion/total_diff.png', difference_2)
