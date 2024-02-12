# Stereo Vision and Point Cloud Generation

This repository contains implementations and results of stereo vision algorithms designed to estimate depth from stereo image pairs and generate corresponding point clouds.

## Repository Structure

- `images/` - Contains the original left, right stereo images used for disparity estimation and the ground truth disparity maps against which algorithm accuracy is evaluated.
- `output/images` - Stores results from the disparity estimations and comparisons.
- `output/meshlab` - Contains results as images of 3D pointclouds rendered in Meshlab.
- `output/point_clouds/` - Features 3D point clouds as XYZ files generated from the disparity maps.

## Algorithms

Included in this repository are two primary algorithms for disparity estimation:

- **Naive Algorithm**: This simple implementation matches pixel blocks from the left and right images to estimate depth.
- **Dynamic Programming Algorithm**: An advanced approach that employs dynamic programming to enhance the accuracy and efficiency of disparity estimation.

### Original Images

<img src="images/art1.png" width="200" alt="art"> <img src="images/books1.png" width="200" alt="books">
<img src="images/dolls1.png" width="200" alt="dolls"> <img src="images/moebius1.png" width="200" alt="reindeer">

### Ground Truth Disparity Maps

<img src="images/art1_true.png" width="200" alt="art"> <img src="images/books1_true.png" width="200" alt="books">
<img src="images/dolls1_true.png" width="200" alt="dolls"> <img src="images/moebius1_true.png" width="200" alt="reindeer">

### Naive Approach Results

<img src="output/images/art_naive.png" width="200" alt="art"> <img src="output/images/books_naive.png" width="200" alt="books">
<img src="output/images/dolls_naive.png" width="200" alt="dolls"> <img src="output/images/moebius_naive.png" width="200" alt="reindeer">

### Dynamic Programming Approach Results

<img src="output/images/art_dp.png" width="200" alt="art"> <img src="output/images/books_dp.png" width="200" alt="books">
<img src="output/images/dolls_dp.png" width="200" alt="dolls"> <img src="output/images/moebius_dp.png" width="200" alt="reindeer">

### Point Cloud Visualizations

<img src="output/meshlab/art_mesh.png" width="200" alt="art"> <img src="output/meshlab/books_mesh.png" width="200" alt="books">
<img src="output/meshlab/dolls_mesh.png" width="200" alt="dolls"> <img src="output/meshlab/reindeer_mesh.png" width="200" alt="reindeer">

See more images and running times in [examples.ipynb](examples.ipynb)

See the algorithms in [stereo/stereo_estimation.py](stereo/stereo_estimation.py)
