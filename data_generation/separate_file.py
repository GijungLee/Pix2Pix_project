import shutil
import os
from natsort import natsorted

file_path = "/Users/gijunglee/code/segemntation/scratch_mask"
files = os.listdir("/Users/gijunglee/code/segemntation/scratch_mask")

os.makedirs("/Users/gijunglee/code/segemntation/scratch_mask2/images", exist_ok=True)
os.makedirs("/Users/gijunglee/code/segemntation/scratch_mask2/masks", exist_ok=True)

for file in natsorted(files):
    if file.split("_")[0] == "mask":
        # print(file)
        shutil.copy2(os.path.join(file_path, file), os.path.join("/Users/gijunglee/code/segemntation/scratch_mask2/masks", file), follow_symlinks=True)
    elif file.split("_")[0] == "org":
        shutil.copy2(os.path.join(file_path, file), os.path.join("/Users/gijunglee/code/segemntation/scratch_mask2/images", file), follow_symlinks=True)