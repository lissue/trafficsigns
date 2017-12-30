# **Traffic Sign Recognition** 

---

**Goals**

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

---

### Dataset Summary & Exploration

#### 1. Summary of the dataset

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the samples for each class of traffic signs are unevenly distributed:

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Image preprocessing

As a first step, I decided to not convert the images to grayscale, because I thought I could use the color info as additional feature other than shape for learning. But I guess a model trained with grayscale would be more robust in the actual world, as color info cannot be guaranteed at low light conditions (or IR).

I then equalized the histogram of the image to enhance the contrast, in hope that it would also help the network to recognize more features more easily.


#### 2. Model architecture

My final model was the LeNet-5 network with slight modification to accommodate the 3 channel RGR images and 43 output classes: ![alt text][image4]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten   	      	| 5x5x16 = 400                     				|
| Fully connected		| 120 outputs  									|
| Dropout       		| 50% dropout rate								|
| Fully connected		| 84 outputs  									|
| Fully connected 		| 43 outputs  									|
| Softmax				| 43 outputs   									|
|						|												|
|						|												|
 


#### 3. Training hyperparameters of the model

To train the model, I used the following for my model.

| Parameter        		|     Value	        					        | 
|:---------------------:|:---------------------------------------------:| 
| Optimizer        		| AdamOptimizer        							| 
| Batch size        	| 128                                        	|
| # epochs				| 15											|
| Learning rate	      	| 0.001                          				|


#### 4. Model performance

My final model results were:
* validation set accuracy of 96%
* test set accuracy of 94%

* Logic behind the chosen architecture

    I started with LeNet-5, because the original application of the network has been for the classification of handwritten characters, thus it seemed to be suitable to classify traffic signs as well. The network as-is already provided 89% validation accuracy when the sample set was fed to the network, without any modifications. With a goal of increasing the validation accuracy by 4%, it seemed reasonable to keep the same architecture and focus on tuning the hyperparameters. (As it turns out, I actually only applied preprocessing, and the model was already able to meet the requirement.)

* Limitations of the initial architecture

    The initial architecture didn't utilize dropout, which seemed to be critical when the samples are less "binary" than the MNIST images.

* Improvements over the initial architecture

    The initial architecture seems to have overfitted the data. Without dropout, the network becomes more dependent on non-repeatable clues found in the training set. This gets worse when the training set population is unevenly distributed among the 43 classes: samples that lack diversity in quantity led to less robust results.
    
    By adding dropout, the network learns to always take repeatability into account, and focus on the traffic sign itself, which is more repeatable than other features that may be present in the image. This can be further enhanced by augmenting the samples in terms of transformations, noise, and motion blur, etc., which is on the to do list.

* Tuning of parameters

    I original increased the number of epochs to 20 from 10. But later found out the last 5 epochs didn't seem to help with improving the accuracy. I'm not sure if further increase the number of epochs would actually help, will need to experiment and find out.

    The training, validation, and test accuracies are in approximate agreement, thus the model seems to be working well. 
 

### Test a Model on New Images

#### 1. Predict using images not included in the training/validation dataset.

Here are five German traffic signs that I found on the web:

![alt text][image3]

The second image might be difficult to classify because the "road work" and "pedestrians" signs are similar visually at a size of 32*32.

#### 2. Performance of model on test images

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

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all images, the model is very certain on the class of the traffic sign (probability of 1).

### Visualization of the outputs from sequential layers

![alt text][image5]

Especially on the first layer, it's apparent that the network focused on the octagon shape of the stop sign, as well the characters on the sign.

### TODO
#### 1. Use greyscale images
#### 2. Augment samples: transformation, noise, motion blur, etc.
