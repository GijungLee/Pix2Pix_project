import os
import cv2
import numpy as np
from natsort import natsorted
import random

# /Users/gijunglee/code/segemntation/data/train_3/masks

img_file_path = "/Users/gijunglee/code/segemntation/data/train_3/images"
img_files = natsorted(os.listdir(img_file_path))
scr_file_path = "/Users/gijunglee/code/segemntation/scratch_mask2 copy/masks/"
scr_files = natsorted(os.listdir(scr_file_path))
dest_path = "/Users/gijunglee/code/segemntation/scratch_mask2 copy/masks2/"
os.makedirs(dest_path, exist_ok=True)

for i, img_file in enumerate(img_files):
    if img_file == ".DS_Store":
        continue
    image = cv2.imread(os.path.join(img_file_path, img_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height = image.shape[0]
    # width = image.shape[1]

    resize_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
    height = resize_image.shape[0]
    width = resize_image.shape[1]

    scratch = cv2.imread(os.path.join(scr_file_path, scr_files[i]))
    scratch = cv2.cvtColor(scratch, cv2.COLOR_BGR2RGB)
    invert = cv2.bitwise_not(scratch)
    # print(invert.shape)
    resize_invert = cv2.resize(invert, (int(height/5), int(width/5)), interpolation=cv2.INTER_AREA)
    # print(resize_invert.shape)
    rand_index1 = random.choice([1.8/4, 2/4, 2.2/4])
    rand_index2 = random.choice([1.8/4, 2/4, 2.2/4])

    # scr_image1 = image[int(height * rand_index1): int(height * rand_index1 + resize_invert.shape[0]),
    #              int(width * rand_index2):int(width * rand_index2 + resize_invert.shape[1])]
    # scr_image2 = image[int(height * rand_index1): int(height * rand_index1 + resize_invert.shape[0]),
    #              int(width * rand_index2):int(width * rand_index2 + resize_invert.shape[1])] + resize_invert
    scr_image1 = resize_image[int(height*rand_index1): int(height*rand_index1+resize_invert.shape[0]),\
                 int(width*rand_index2):int(width*rand_index2+resize_invert.shape[1])]
    scr_image2 = resize_image[int(height*rand_index1): int(height*rand_index1+resize_invert.shape[0]),\
                 int(width*rand_index2):int(width*rand_index2+resize_invert.shape[1])] + resize_invert
    # scratch_image = image.copy()
    scratch_image = resize_image.copy()
    scratch_image[int(height*rand_index1): int(height*rand_index1+resize_invert.shape[0]),
                 int(width*rand_index2):int(width*rand_index2+resize_invert.shape[1])] = scr_image2
    # print(scr_image1.shape)
    # print(scr_image2.shape)
    # break
    cv2.imshow("ddd", invert)
    cv2.imshow("dd", resize_invert)
    # cv2.imshow("image", image)
    cv2.imshow("image", resize_image)
    cv2.imshow("Invert", scr_image1)
    cv2.imshow("ii", scr_image2)
    cv2.imshow("dddd", scratch_image)
    # cv2.imshow("invert", scr_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
