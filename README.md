# 3DSSF-Stereo-Matching

Implemented a Naive and a Dynamic Programming stereo matching scheme.

**Input examples**

<img src="images/art1.png" width="200" alt="art"> <img src="images/books1.png" width="200" alt="books">
<img src="images/dolls1.png" width="200" alt="dolls"> <img src="images/moebius1.png" width="200" alt="reindeer">

**Ground truth**

<img src="images/art1_true.png" width="200" alt="art"> <img src="images/books1_true.png" width="200" alt="books">
<img src="images/dolls1_true.png" width="200" alt="dolls"> <img src="images/moebius1_true.png" width="200" alt="reindeer">

**Naive disparities**

<img src="output/images/art_naive.png" width="200" alt="art"> <img src="output/images/books_naive.png" width="200" alt="books">
<img src="output/images/dolls_naive.png" width="200" alt="dolls"> <img src="output/images/moebius_naive.png" width="200" alt="reindeer">

**Dynamic Programming disparities**

<img src="output/images/art_dp.png" width="200" alt="art"> <img src="output/images/books_dp.png" width="200" alt="books">
<img src="output/images/dolls_dp.png" width="200" alt="dolls"> <img src="output/images/moebius_dp.png" width="200" alt="reindeer">

**Point clouds**

<img src="output/meshlab/art_mesh.png" width="200" alt="art"> <img src="output/meshlab/books_mesh.png" width="200" alt="books">
<img src="output/meshlab/dolls_mesh.png" width="200" alt="dolls"> <img src="output/meshlab/reindeer_mesh.png" width="200" alt="reindeer">

See more images and running times in [examples.ipynb](examples.ipynb)
See the algorithms in [stereo/stereo_estimation.py](stereo/stereo_estimation.py)
