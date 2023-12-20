import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/0.jpg')

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.show()