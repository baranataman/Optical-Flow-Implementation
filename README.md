# Optical Flow Implementation
Face Feature Tracker with Kanade-Lucas-Tomasi (KLT) Optical Flow Algorithm Implementation in Python

![optical_flow](https://user-images.githubusercontent.com/23663934/183895583-5aab03a4-8157-4524-8ba3-6574c4cc0e8b.gif)

Main steps of the algorithm:
1) The very first frame is captured and the face is detected (line 47)

2) Then the trackable corner points on the face are extracted (line 59). This is achieved by off-the-shelf OpenCV library.

3) Following the feature detection, those **local** feature points are converted to **global** points on the whole frame (line 68-69). Then those **global** points are marked on the video.

4)  As there are enough feature points to track, OpenCV's built-in Lucas-Kanade Optical Flow algorithm is hired to track the feature points (line 79). The parameters are tailored for the optimum result by trial and error.  

5) At the end, the previous frame is assigned as the current frame to be traced in the next iteration.
