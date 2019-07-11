# General Image Encoding
Using AutoML and unsupervised learning to generate images from non-image data.

## Summary
This project aims to encode any tabular data into image data. The way this goal is achieved is fairly simple, and follows these steps
  * Make a ton of synthetic features from the data; enough such that the total number of features is a perfect square (preferably like 256 or 1,024 columns).
  * Reassign the columns such that they would represent spots of similarity in an image (i.e. create an image of features, where similar features are closer together).
From here, we could use an image classifier/regressor to perform supervised learning on the image.
