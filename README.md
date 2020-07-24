# Level 2-3 Autonomous Driving System - Using Carla #

- orbslam.py - toy implementation of monocular SLAM (ORBSlam) - INCOMPLETE
- model.py - Neural Network Model - architecture needs improvement
- renderer.py - Radar point cloud visualization

## TODO: ##
- Research on different ANN Architectures to use - currently thinking of using a VAE and Inception CNN with 
residual connections.
- Implement LSTM after flattening (cnnLSTM)
- Set up data trigger infrastructure - data collection for edge cases
- Get at least a toy implementation of ORBSlam up and running

# Goal:
My final goal is to get a Level 2-3 Self-Driving System up and running as soon as possible (will take some time)

# NOTES #
## Multiple View Geometry in Computer Vision: ##
### Notation used in book: ###
 - A bold-face symbol such as **x** always
   represents a column vector, and its transpose is the row vector **x**<sup>T<sup>.
 - a point in the plane will be represented by the column vector
   _(x,y)_<sup>T<sup>, rather than its transpose, the row vector _(x,y)_.
 - a line may naturally be represented by the vector _(a, b, c)_<sup>T<sup>
 - The set of equivalence classes of vectors in **R**<sup>3<sup> − (0, 0, 0)T forms the projective
   space **P**<sup>2<sup>. 
 - A conic is a curve described by a second-degree equation in the plane. In Euclidean
   geometry conics are of three main types: hyperbola, ellipse, and parabola
 - The equation of a conic in inhomogeneous coordinates is:
   _ax<sup>2<sup> + bxy + cy<sup>2<sup> + dx + ey + f = 0_
 - i.e. a polynomial of degree 2. “Homogenizing” this by the replacements:
   x → x1/x3, y → x2/x3 gives
   ax<sub>1<sub><sup>2<sup> + bx<sub>1<sub>x<sub>2<sub> + cx<sub>2<sub><sup>2<sup> + dx<sub>1<sub>x<sup>3<sup> + ex<sub>2<sub>x<sub>3<sub> + fx<sub>3<sub><sup>2<sup> = 0 (2.1)
   or in matrix form:
   **x**<sup>T<sup>C**x** = 0 
