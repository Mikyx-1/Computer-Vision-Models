# Implementation of Spatial Transformer Network 

## 1. What is Spatial Transformer Network ?
A Network that helps us to retain geometric information. In contrast to Convolutional layers, fully connected layers they tend to distort or change the geometry of the input.

For example: When you rotate the image 90 degrees, the directions have changed but the geometric information still remains. Unfortunately, for CNN, the rotated image will have the CNN output differently. 

![Screenshot from 2023-12-31 16-34-11](https://github.com/Mikyx-1/Computer-Vision-Models/assets/92131994/06c6530b-d8fe-4877-8cc6-8bb586f364c0)


## 2. How does it work ?
1st: The localisation net will find out the transformation matrix 

2nd: Apply the transformation matrix to the image to find the mapping pairs

3rd: Based on the mapping pairs, the sampler will take the target values from source and map it to the output image

4th: Output the geometricall transformed image  

So if you rotate the image, flip it. **The output of the STN is still close to the original image**. That's the point

Note: Im the image, the sampler and the grid generators work in parallel to speed up computations

![image](https://github.com/Mikyx-1/Computer-Vision-Models/assets/92131994/8c236b93-b3b0-406d-86da-9aa5971f3bfe)

## 3. Applications
If you dont want your input image to be distorted (for example: 3D points classification, 3D points segmentation, ...)
 
P/s: To gain more insights, please visit this link: https://towardsdatascience.com/spatial-transformer-tutorial-part-1-forward-and-reverse-mapping-8d3f66375bf5 
