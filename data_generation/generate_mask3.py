import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# file_path = "/Users/gijunglee/code/segemntation/untitled folder 5/masks"
# file_path = "/Users/gijunglee/code/segemntation/data/project-2-at-2022-05-20-03-48-ae9a8e4f"
# file_path = "/Users/gijunglee/code/segemntation/untitled folder 4/project-2-at-2022-05-20-03-48-ae9a8e4f"
# file_path = "/Users/gijunglee/Downloads/project-1-at-2022-05-22-20-17-15d4e9f8"
# file_path = "/Users/gijunglee/Downloads/project-1-at-2022-06-06-02-27-cb2b3c14"
file_path = "/Users/gijunglee/Downloads/project-11-at-2022-06-07-05-51-52dbd7ef"
# files = os.listdir("project-1-at-2022-04-29-01-30-da164206")

if os.path.exists(os.path.join(file_path,".DS_Store")):
    os.remove(os.path.join(file_path,".DS_Store"))
    print("The file has been deleted successfully")
else:
    print("The file does not exist!")

if "test_data25" not in os.listdir("/Users/gijunglee/code/segemntation"):
    os.makedirs("/Users/gijunglee/code/segemntation/test_data25/masks")
files = os.listdir(file_path)
files.sort()
print(files)
count = 251
image = []

for i, file in enumerate(files):
    if i == 0:
        file1 = cv2.imread(os.path.join(file_path, file))
        file1 = cv2.cvtColor(file1, cv2.COLOR_BGR2RGB)
        shape = np.shape(file1)  # print(shape)
        # # print(shape)
        image = np.zeros(shape)
        # print(image.shape)

    # count += 1
    file_number = files[i].split("-")[1]
    if file_number != files[i-1].split("-")[1] and i != 0:
        cv2.imwrite(f"/Users/gijunglee/code/segemntation/test_data25/masks/mask{count}.jpg", image)
        count += 1

        file1 = cv2.imread(os.path.join(file_path, file))
        file1 = cv2.cvtColor(file1, cv2.COLOR_BGR2RGB)
        shape = np.shape(file1)
        image = np.zeros(shape)
        # count += 1
    # shape = np.shape(file1)  # print(shape)
    # # print(shape)
    # image = np.zeros(shape)
    # print(image.shape)
    else:
        file1 = cv2.imread(os.path.join(file_path, file))
        file1 = cv2.cvtColor(file1, cv2.COLOR_BGR2RGB)

    if file.split("-")[7] == "1":
        file1 = file1 * (0.1, 0.1, 0.1) # black
    elif file.split("-")[7] == "4":
        file1 = file1 * (1, 0, 0) # blue
    elif file.split("-")[7] == "5":
        file1 = file1 * (0, 1, 0) # green
    elif file.split("-")[7] == "6":
        file1 = file1 * (0.5, 0, 0.5) # purple
    elif file.split("-")[7] == "7":
        # file1 = file1 * (0.52, 0.82, 1) # yellow
        file1 = file1 * (0, 0.3, 0.5)
    # pin
    elif file.split("-")[7] == "0":
        file1 = file1 * (0.25, 0.25, 0.25)  # gray
    # scratch
    elif file.split("-")[7] == "2":
        file1 = file1 * (1, 1, 1)  # white
    # marking
    elif file.split("-")[7] == "3":
        file1 = file1 * (0.8, 0.1, 0.8)  # less white
    elif file.split("-")[7] == "8":
        file1 = file1 * (0.3, 0.6, 0.6)
    image += file1
cv2.imwrite(f"/Users/gijunglee/code/segemntation/test_data25/masks/mask{count}.jpg", image)
