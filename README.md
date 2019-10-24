# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-nvidia.png "Model Architecture"
[image2]: ./examples/summary.png "Model Summary"
[image3]: ./examples/right.jpg "Image Sample"
[image4]: ./examples/right_flipped.jpg "Flipped Image Sample"
[image5]: ./examples/loss.jpg "Number of Epoch"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
#### 1. An appropriate model architecture has been employed

I adapted the NVIDIA model (with some minor modifications) as shown in the figure below:
![alt text][image1]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input - Normalized    | 160x320x3    				 					|
| Input - Cropped       | 90x300x3 - cropped using Cropping2D  			| 
| Convolution 5x5     	| 2x2 stride, depth = 24, activation= relu		|
| Convolution 5x5     	| 2x2 stride, depth = 36, activation= relu		|
| Convolution 5x5     	| 2x2 stride, depth = 48, activation= relu		|
| Convolution 3x3     	| 1x1 stride, depth = 64, activation= relu		|
| Convolution 3x3     	| 1x1 stride, depth = 64, activation= relu		|
| Fully connected 01	| output = 100 				    				|
| Dropout   	      	| Using `tf.nn.dropout()`, Keep_prob = 0.3  	|
| Fully connected 02	| output = 50 				    				|
| Fully connected 03	| output = 10 				    				|
| Fully connected 04	| output = 1 				    				|

In my code (model.py) I created the function `nvidia_model()` defining the model (lines 88-113). As it can be noticed, I have added a dropout layer to reduce the effect of overfitting. I have also introduced a 'relu' activation function to the five convolutional layers to introduce some non-linearity to the model. It is also important to mention that the output of the last layer is 1 because we are dealing with a regression problem.

Below is a summary of the model provided by Keras:

![alt text][image2]

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer after the first fully connected layer in order to reduce overfitting (model.py line 107). I didn’t introduce more dropout layers because one layer was just sufficient to generate acceptable results. The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 123-133). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. To be able to use the right and left cameras images, I had to adjust the steering angle to resemble the images from the vehicle's center point of view. For the right images, I subtracted 0.2 from the original steering angle and for the left image, I added 0.2 to the original steering angle.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try to mimic or use a well-known model that earned its good reputation in the field of self-driving vehicles. As I discussed before, I adapted the NVIDIA model as it was recommended in the lectures.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I modified the model by including a dropout layer after the first fully connected layer with keeping probability of 0.3. Then I reduce the number of epochs from 5 to 3 because the training and validation errors stopped decreasing.    
The final step was to run the simulator to see how well the car was driving around track one. It was driving probably without drifting to the sides of the road.   At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-113) consisted of a convolution neural network with the layers and layer sizes as described previously.

The main functions in (model.py) are:
* `reading_data()`: A function that reads the data properly (as discussed before) and returns an array of the paths of the images and an array for the labels.
* `generator()`: A function that takes as an input the array of the image paths and the labels and returns the colored RGB images and their corresponding labels in batches that are loaded to the memory on demand. I used a batch size of 32.
* `nvidia_model()`: A function that returns the modified NVIDIA model.


#### 3. Creation of the Training Set & Training Process

For training and validating the model I used the provided data in `driving_log.csv`, but that was not enough, because since for track 1 most the curves are left curves, the model would be biased to left curves and wouldn’t perform appropriately when it encounters a right curve. To mitigate this issue, I augmented data by flipping all the original images (model.py lines 70) with an associated label that is the inverses of the original label. here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

After the data augmentation, I had a number of 24,108 images. I then shuffled the data and split it into a training and validation sets (80% and 20% respectively) to have 19,286 images in the training set and 4,822 images in the validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the image below. I used an adam optimizer so that manually training the learning rate wasn't necessary. The training and validation losses from each epoch are shown in the table below:

![alt text][image5]

| Epochs         		|     Training Error   	|   Validation Error   	| 
|:---------------------:|:---------------------:|:---------------------:| 
| 01				    | 0.0185    			| 0.0156    			|
| 02				    | 0.0164    			| 0.0148    			| 
| 03				    | 0.0154    			| 0.0144    			|


