import face_recognition # for face recognition
import imutils          # for getting paths of file
import pickle           # for serializing and de-serializing a Python object structure
import time             # for system waiting
import cv2              # for computer vision problem
import os               # for finding path



def face_unlock():
    # find path of xml file containing haarcascade file 
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read()) # de-serializing the file and read the face data
     
    print("Face unlock is starting. Please look at the camera.")
    
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Start video capturing

    name = "Unknown"
    for i in range (50):
        print("Scanning for the",i+1,"/ 50 times")
        
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read() # get the returned value and frame
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the frame from BGR to GRAY
        except:
            pass
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        
        try:
            # convert the input frame from BGR to RGB 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            pass
        
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []
        
        # loop over the facial embeddings in case
        # we have multiple embeddings for multiple faces
        for encoding in encodings:
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely and False for rest
            matches = face_recognition.compare_faces(data["encodings"], encoding)

            # check to see if we have found a match
            if True in matches:
                # Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for each recognized face
                for i in matchedIdxs:
                    # Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    
                    # increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                    
                #set name which has highest count
                name = max(counts, key=counts.get)
                
                video_capture.release()
                cv2.destroyAllWindows()
                return name
            
    video_capture.release()
    cv2.destroyAllWindows()
    return name



name=face_unlock()
   
if name=='Unknown':
    print('Identity not found!')
else:
    print('Identity confirmed:',name)
    
end=input('\nPress enter to terminate the program...')
