import cv2
from random import randrange
# Download a bunch of pictures
# Change the color
# Train your algorithm to identify the monochrome picture you've created
#Steps in Face Detection
#1. You get a Classifier(for the specific type of classisfication you want to do)
#2. Get the image you need and convert it to gray-scaled (i.e black and white)
#3. Put your image in the classifier to be trained by the classifier in detecting the face  
#Classifiers for image detection
    #[Frontal Face Detection]
trainedFaceData = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
#Choosing an image for the algorithm to classify(reading it)
img = cv2.imread('Images/RU1.webp')
multiFace = cv2.imread('Images/multiFrontalFace1.jpg')
#Choosing a video for the classifier to the detect the moving image
webcam = cv2.VideoCapture(0)
#Iterate over frames
while True:
    #Read the current frame
    successfulFrameRead, frame = webcam.read()
    #Gray Scaled Video(Remeber the classifer only recognises gray sacled image[video])
    grayScaledVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect Faces
    videoCoordinates = trainedFaceData.detectMultiScale(grayScaledVideo)
    #Draw Rectangles on the face
    for (x,y,w,h) in videoCoordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
    #Displaying the image[video]
    cv2.imshow('Image[Video] Detector', frame)
    key = cv2.waitKey(1)
    #Stop if Q is pressed
    if key==81 or key==113:
        break
#Release the VideoCapture object
video.release() 
#Coverting to gray scale
grayScaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayScaledMultiFace = cv2.cvtColor(multiFace, cv2.COLOR_BGR2GRAY)
#Detecting Faces (Coordinates)
imgCoordinates = trainedFaceData.detectMultiScale(grayScaledImg)
multiFaceCoordinates = trainedFaceData.detectMultiScale(grayScaledMultiFace)
#Draw Rectangles around the face(img, (x,y), (x+w,y+h), (color), line width)
#cv2.rectangle(img, (105,49), (105+175, 49+175), (0,0,0), 1)
(x,y,w,h) = imgCoordinates[0]
cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(256)), 1)
#Looping through faces in an image[Multi-Faced Image]
for (x,y,w,h) in multiFaceCoordinates:
    cv2.rectangle(multiFace, (x,y), (x+w, x+h), (randrange(256),randrange(256),randrange(256)), 2)
#Displaying the classified image
cv2.imshow('Face Detector', img)
cv2.imshow('Multi Faces', multiFace)
#Coordinates
print('Image Coordinates - ',imgCoordinates)
print('Multi-Face Coordinates - ',multiFaceCoordinates)

cv2.waitKey()
print('Code Completed')