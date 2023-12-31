# Implementation of Spatial Transformer Network 

## 1. What is Spatial Transformer Network ?
A Network that helps us to retain geometric information. In contrast to Convolutional layers, fully connected layers they tend to distort or change the geometry of the input.
For example: When you rotate the image 90 degrees, the directions have changed but the geometric information still remains. Unfortunately, for CNN, the rotated image will have the CNN output differently. 

## 2. How does it work ?
1st: The localisation net will find out the transformation matrix 
2nd: Apply the transformation matrix to the image to find the mapping pairs
3rd: Based on the mapping pairs, the sampler will take the target values from source and map it to the output image
4th: Output the geometricall transformed image  
Note: Im the image, the sampler and the grid generators work in parallel to speed up computations
![image](https://github.com/Mikyx-1/Computer-Vision-Models/assets/92131994/8c236b93-b3b0-406d-86da-9aa5971f3bfe)

## 3. Applications
Point Net
P/s: 
