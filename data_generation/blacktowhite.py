import cv2
import os
from natsort import natsorted


file_path = "/Users/gijunglee/code/segemntation/scratch_mask2 copy/masks/"
files = natsorted(os.listdir(file_path))
dest_path = "/Users/gijunglee/code/segemntation/scratch_mask2 copy/masks2/"
os.makedirs(dest_path, exist_ok=True)
print(files)
for file in files:
    image = cv2.imread(os.path.join(file_path, file), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    invert = cv2.bitwise_not(image)
    invert = invert * (0.25, 0.25, 0.25)
    cv2.imwrite(os.path.join(dest_path, file), invert)

# image = cv2.imread(file_path+"resistors_320.png", 1)
# print(image.shape)
# invert = cv2.bitwise_not(image)
# cv2.imshow("Original", image)
# cv2.imshow("Invert", invert)
# cv2.waitKey(0)
# cv2.destroyAllWindows()