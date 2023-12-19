import cv2

haar_cascade = 'input/haarcascades/cars.xml'
video = 'cartest.mp4'
      
#car_cascade = cv2.CascadeClassifier(haar_cascade)


# loop runs if capturing has been initialized. 
cap = cv2.VideoCapture(video) 
while cap.isOpened(): 
    # reads frames from a video 
    print('loop')
    ret, frames = cap.read() 
    if ret == False:
        print('Error ret false')
        
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
        
    
    # Detects cars of different sizes in the input image 
    #cars = car_cascade.detectMultiScale(gray, 1.1, 1) 
        
    # To draw a rectangle in each cars 
    #for (x,y,w,h) in cars: 
        #cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
    
    # Display frames in a window  
    cv2.imshow('video', frames) 
        
    # Wait for Esc key to stop 
    if cv2.waitKey(33) == 27: 
        break
    
# De-allocate any associated memory usage 
cv2.destroyAllWindows()