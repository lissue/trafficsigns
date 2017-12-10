# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample_overview.png "Visualization"
[image2]: ./sample_dist.png "Distribution"
[image3]: ./testimages.png "Test images"
[image4]: https://github.com/udacity/CarND-LeNet-Lab/raw/master/lenet.png "LeNet-5"
[image5]: ./featuremap.png "Output Feature Map"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lissue/trafficsigns/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the samples for each class of traffic signs are unevenly distributed:

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to not convert the images to grayscale, because I thought I could use the color info as additional feature other than shape for learning. But I guess a model trained with grayscale would be more robust in the actual world, as color info cannot be guaranteed at low light conditions (or IR).

I then equalized the histogram of the image to enhance the contrast, in hope that it would also help the network to recognize more features more easily.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was the LeNet-5 network with slight modification to accommodate the 3 channel RGR images and 43 output classes: ![alt text][image4]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| Relu                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten   	      	| 5x5x16 = 400                     				|
| Fully connected		| 120 outputs  									|
| Dropout       		| 50% dropout rate								|
| Fully connected		| 84 outputs  									|
| Fully connected 		| 43 outputs  									|
| Softmax				| 43 outputs   									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....
| Parameter        		|     Value	        					        | 
|:---------------------:|:---------------------------------------------:| 
| Optimizer        		| AdamOptimizer        							| 
| Batch size        	| 128                                        	|
| # epochs				| 20											|
| Learning rate	      	| 0.001                          				|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96%
* test set accuracy of 94%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    I started with LeNet-5 as it already provided 89% validation accuracy when the sample set was fed to the network, without any modifications. With a goal of increasing the validation accuracy by 4%, it seemed reasonable to keep the same architecture and focus on tuning the hyperparameters. (As it turns out, I actually only applied preprocessing, and the model was already able to meet the requirement.)
* What were some problems with the initial architecture?
    The initial architecture didn't utilize dropout, which seemed to be critical when the samples are less "binary" than the MNIST images.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    The initial architecture seems to have overfitted the data. Without dropout, the network becomes more dependent on non-repeatable clues found in the training set. This gets worse when the training set population is unevenly distributed among the 43 classes: samples that lack diversity in quantity led to less robust results.
    By adding dropout, the network learns to always take repeatability into account, and focus on the traffic sign itself, which is more repeatable than other features that may be present in the image. This can be further enhanced by augmenting the samples in terms of transformations, noise, and motion blur, etc., which is on the to do list.
* Which parameters were tuned? How were they adjusted and why?
    I original increased the number of epochs to 20 from 10. But later found out the last 5 epochs didn't seem to help with improving the .
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    As stated above.

If a well known architecture was chosen:
* What architecture was chosen?
    I used the LeNet-5 network.
* Why did you believe it would be relevant to the traffic sign application?
    The original application of the network has been for the classification of handwritten characters, thus it seemed to be suitable to classify traffic signs as well.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The training, validation, and test accuracies are in approximate agreement, thus the model seems to be working well. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]

The second image might be difficult to classify because the "road work" and "pedestrians" signs are similar visually at a size of 32*32.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Children crossing		| Children crossing 							|
| 50km/h				| 50km/h										|
| 60km/h          		| 60km/h    					 				|
| 30km/h        		| 30km/h              							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

The images are relatively clean shot of the traffic signs, which didn't have much blur, and helped with the high test accuracy. Augmenting the training sample, and adding more noisy images to the testing is on the to do list.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all images, the model is very certain on the class of the traffic sign (probability of 1).

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
![alt text][image5]
Especially on the first layer, it's apparent that the network focused on the octagon shape of the stop sign, as well the characters on the sign.

### TODO
#### 1. Use greyscale images
#### 2. Augment samples: transformation, noise, motion blur, etc.
