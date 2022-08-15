import os
import glob
from natsort import natsorted

count = 176
# start = 1

# path = "/Users/gijunglee/code/segemntation/test_data16/images"
# path = "/Users/gijunglee/code/segemntation/untitled folder 4/scratch_images"
# path = "/Users/gijunglee/code/segemntation/untitled folder 7"
path = "/Users/gijunglee/code/segemntation/data/train_12 copy/untitled folder 2"
files = natsorted(glob.glob(path + "/*.png"))
# files = natsorted(glob.glob(path + "/*.jpg"))
print(files)
for i, file in enumerate(files):
    old_name = file
    # new_name = path + f"/task-{count}-annotation-2-by-1-tag-2-0.png"
    new_name = path + f"/resistors_{count}.png"
    # new_name = path + f"/mask{count}.jpg"
    os.rename(old_name, new_name)
    count += 1
