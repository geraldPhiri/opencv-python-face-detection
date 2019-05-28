import cv2
import numpy #will help us represent our images(frames)

#create a classifier to help us in finding faces
classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#get Camera
cam=cv2.VideoCapture(0)

retrieved, frame=cam.read()

#if a frame has been read loop
while(retrieved):
 
    #find faces in grayscale frame remember that grayscale
    #is 2-dimensional thus it contains less computaional information
    #thus it makes the process of finding faces quicker
    faces=classifier.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),1.3,5) #Try changing 1.3 and 5. based on the values you use 
                                                                                    #the program might become better at finding faces or worse
    
    #draw faces on the color frame or 3-dimensional image
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#(255,0,0) is for color of rectangle, 2 is for width of rectangle
    cv2.imshow("face detection(python+opencv)",frame)
    cv2.waitKey(2)
    retrieved, frame=cam.read()
cam.release()
