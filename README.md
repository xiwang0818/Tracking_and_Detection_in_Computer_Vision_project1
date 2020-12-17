# Tracking_and_Detection_in_Computer_Vision_project1
This is one of my project in my Intern ,Tracking and Detection in Computer Vision, at Chair for Computer Aided Medical Procedures & Augmented Reality, Technical University of Munich.  
Task1 is to prepare model and extract SIFT keypoints. In this task I visualized the model, the camera postion and also the SIFT keypoints in one 3D space.  
Task2 is to realize the pose estimation with PnP and RANSAC.  
Task3, different from previous task, is to do the pose refinement with non-linear optimization method(IRLS Algorithm-----Iteratively Re-weighted Least Squares).  
This method is a robust version of the Levenberg-Marquart(LM) optimization algorithm, which itself is a mixture of the Gauss-Newton method(GN) and simple gradient descent. Firstly I initialize the rotation and translation vector of the first image using PnP-Ransac in task2. I secondly project the matched 3D SIFT points from previous image to current 2D coordinate. Thirdly, I did the non-linear interative optimization until it converges.
