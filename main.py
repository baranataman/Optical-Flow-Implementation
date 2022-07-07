import cv2
import time
import traceback
import numpy as np
from matplotlib import pyplot as plt

def get_delay(start_time, fps=30):
    if (time.time() - start_time) > (1 / float(fps)):
        return 1
    else:
        return max(int((1 / float(fps)) * 1000 - (time.time() - start) * 1000), 1)


# Cascade classifier to etect human faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start the camera capture
cam = cv2.VideoCapture(0)


gray_prev = None  # previous frame assigned as null at the beginning
p0 = []  # previous point array

while True:
    start = time.time()
              
    # Get athe first frame
    ret_val, img = cam.read()
    if not ret_val:
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
        gray_prev = None  # previous frame
        p0 = []  # previous point array
        continue

    else:
        # Mirror the frame and convert to grayscale
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        minimum_feature_point = 10 # variable
        if len(p0) <= minimum_feature_point:  # if there are less than 10 feature points
            # Detect the feature points
            img = cv2.putText(img, 'Detection', (0,20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,255,255))

            # Detect faces with off-the-shelf library method of face_cascade
            faces = face_cascade.detectMultiScale(gray, 
                             scaleFactor=1.3, 
                             minNeighbors=4, 
                             minSize=(30, 30),
                             flags=cv2.CASCADE_SCALE_IMAGE)
            
            # Take the first face and get trackable points.
            if len(faces) != 0:
                # Region of Interest 
                roi_gray = gray[faces[0,0]+5:faces[0,0]+faces[0,2]-25, faces[0,1]+60:30+faces[0,1]+faces[0,3]]

                # Get trackable points from this region of interest with OpenCV off-the-shelf function
                p0 = cv2.goodFeaturesToTrack(roi_gray, 
                                             maxCorners=70,
                                             qualityLevel=0.001,
                                             minDistance=5)

                # Convert points to form (point_id, coordinates)
                p0 = p0[:, 0, :] if p0 is not None else []
                
                # Convert from ROI to image coordinates
                p0[:,0] = p0[:,0] + faces[0,0]
                p0[:,1] = p0[:,1] + faces[0,1]

            # Save grayscale copy for next iteration
            gray_prev = gray.copy()
            
        else:
            # Now if there are enough feature points, do tracking
            img = cv2.putText(img, 'Tracking', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0, 255, 255))

            # Calculate optical flow using calcOpticalFlowPyrLK of OpenCV
            p1, isFound, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, 
                                                        None,
                                                        winSize=(31,31),
                                                        maxLevel=10,
                                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
                                                        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                                                        minEigThreshold=0.00025)
           
            aa = isFound>0
            points = p1[aa[:,0],:]
            
            # Draw feature points using OpenCV drawMarker
            for i in range(0,points.shape[0]):
                img = cv2.drawMarker(img, points[i,:].astype(np.int32), color=(0,255,0))
            
            
            # Update points and the previous frame 
            p0 = points
            gray_prev = gray.copy()

        # Quit text
        img = cv2.putText(img, 'Press q to quit', (440, 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
        cv2.imshow('Video feed', img)

    # Limit FPS to ~30
    if cv2.waitKey(get_delay(start, fps=30)) & 0xFF == ord('q'):
        break  # q to quit
        
   
            
# Close camera and video feed window
cam.release()   
cv2.destroyAllWindows()

