
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/0.jpg')

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.show()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')

#edges = cv2.Canny(gray_img,15,150,0)
# blur = cv2.medianBlur(gray_img, 3)
# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# sharpen = cv2.filter2D(gray_img, -1, sharpen_kernel)

detected_edges = cv2.Canny(gray_img,9, 150, 3)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

#dil = cv2.dilate(detected_edges,kernel,iterations = 1)
close = cv2.morphologyEx(detected_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

#close = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)

#ret, close = cv2.threshold(close, 20, 255, cv2.THRESH_BINARY)

circles = cv2.HoughCircles(close,cv2.HOUGH_GRADIENT,1.1,20,param1=50,param2=30,minRadius=5,maxRadius=55)
print(circles)

plt.imshow(close, cmap='gray')

circles=circles[0,:]
print(circles)

for i in circles:
    # draw the outer circle
    cv2.circle(rgb_img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    # draw the center of the circle
    cv2.circle(rgb_img,(int(i[0]),int(i[1])),2,(0,0,255),3)

print(len(circles))
plt.imshow(rgb_img)
plt.show()

contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))
print((hierarchy[0]))

x0, y0, w0, h0= cv2.boundingRect(contours[0])
cv2.rectangle(rgb_img, (x0,y0),(x0+w0,y0+h0), (0,255,0),5)

plt.imshow(rgb_img)

dice0 = close[y0:y0+h0, x0:x0+w0]

plt.imshow(dice0, cmap='gray')
plt.show()


circles0 = cv2.HoughCircles(dice0,cv2.HOUGH_GRADIENT,1.3,20,param1=50,param2=30,minRadius=5,maxRadius=55)
print(len(circles0[0]))

cv2.putText(rgb_img, f'score: {len(circles0[0])}', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
plt.imshow(rgb_img)
plt.show()

