# Level 2-3 Autonomous Driving System - Using Carla #

- orbslam.py - toy implementation of monocular SLAM (ORBSlam) - INCOMPLETE
- model.py - Neural Network Model - architecture needs improvement, is currently
  just a basic ResNet
- renderer.py - Radar point cloud visualization, very slow lol

## TODO: ##
- Machine Learning
	- Research on different ANN Architectures 
		- Implement recurrence, temporal dynamics of scene
		- Semantic Segmentation
	- Look more into automatic labelling of moving vs. non-moving objects in seen (have my own little idea wanna
	  implement :smile:).
	- Set up data trigger infrastructure - data collection for edge cases

- SLAM
	- Read more of Multiple View Geometry
	- Figure out what to do after getting Camera Matrix (projection matrix?)
	- Get at least a toy implementation of ORBSlam up and running

- Misc.
	- Leverage Cloud GPU for training and stuff (Google Colab)
	- git gud


# Goal:
My final goal is to get a Level 2-3 Self-Driving System up and running, whilst also learning about relevant concepts

# NOTES #
## Multiple View Geometry in Computer Vision: ##
### Notation used in book: ###
 - A bold-face symbol such as **x** always
   represents a column vector, and its transpose is the row vector **x**<sup>T</sup>.
 - a point in the plane will be represented by the column vector
   _(x,y)_<sup>T</sup>, rather than its transpose, the row vector _(x,y)_.
 - a line may naturally be represented by the vector _(a, b, c)_<sup>T</sup>
 - The set of equivalence classes of vectors in **R**<sup>3</sup> − (0, 0, 0)T forms the projective
   space **P**<sup>2</sup>. 
 - A conic is a curve described by a second-degree equation in the plane. In Euclidean
   geometry conics are of three main types: hyperbola, ellipse, and parabola
 - The equation of a conic in inhomogeneous coordinates is:
   _ax<sup>2</sup> + bxy + cy<sup>2</sup> + dx + ey + f = 0_
 - i.e. a polynomial of degree 2. “Homogenizing” this by the replacements:
   x → x1/x3, y → x2/x3 gives
   ax<sub>1</sub><sup>2</sup> + bx<sub>1</sub>x<sub>2</sub> + cx<sub>2</sub><sup>2</sup> + dx<sub>1</sub>x<sup>3</sup> + ex<sub>2</sub>x<sub>3</sub> + fx<sub>3</sub><sup>2</sup> = 0 (2.1)
   or in matrix form:
   **x**<sup>T</sup>C**x** = 0 

## Overview of Steps for implementing ORBSlam: #
 1. Get two images points, first image being **x** and next one being **x**'
 2. Extract features(keypoints) from  each image _u,v_, along with descriptors
	of every point using ORB.
 3. Use KNN with point descriptors to match points from each image
 4. Filter out points using ratio test (distance from points) 
 5. Extract the Fundamental matrix **F**, using RANSAC to filter matches and the 8-point algorithm.
 6. Perform SVD on **F**, and find focal length values from s<sub>1</sub> and s<sub>1</sub> of matrix **S** (need to explore different method, currently unsure)
 7. Set the Intrinsic Matrix **K** using focal length and center point of pinhole (need to investigate further on using better methods, extracting focal length and getting intrinsic params.).
 8. Repeat steps from 1-4, then normalize homogenous image points _(x,y) -> (x,y,z)_ **x** and **x**' into **x^^** and **x^^**':

    **x^^** = **K**<sup>-1</sup>**x**

    **x^^**' = **K**<sup>-1</sup>**x**'
 9. Find the Essential Matrix **E* *using _(x,y)__ from normalized coords 
 10. Perform Singular Value Decomposition on **E**
 11. Extract Rotation and translation as demonstrated here:
     [Determining R and t from E](https://en.wikipedia.org/wiki/Essential_matrix#Determining_R_and_t_from_E)
 12. And finally, extract 3D points as shown here:
     [3D points from image points](https://en.wikipedia.org/wiki/Essential_matrix#3D_points_from_corresponding_image_points)
