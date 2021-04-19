from imutils import paths   # get paths of files in a folder
import face_recognition     # for face recognition
import pickle               # for serializing and de-serializing a Python object structure
import cv2                  # for computer vision problem
import os                   # for opening and creating files in os
 
# get paths of each file in folder named 'faces'
# Images here contains folders of people
imagePaths = list(paths.list_images('faces')) # create a list that contains the paths of every image in the folder 'faces'
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    
    # extract the person name from the image path (folder name)
    name = imagePath.split(os.path.sep)[-2]
    
    # load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB
    
    # Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')
    
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        
# save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}

# use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()
