# **Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Model Visualization"
[image2]: ./examples/center_example.jpg "Center Image"
[image3]: ./examples/right_recover.jpg "Recovery Right Image"
[image4]: ./examples/left_recover.jpg "Recovery Left Image"
[image5]: ./examples/jungle_example.jpg "Jungle Example"
[image6]: ./examples/mountain_example.jpg "Mountain Example"
[image7]: ./examples/original_turn.jpg "Original Turn Image"
[image8]: ./examples/flipped_turn.jpg "Flipped Turn Image"
[image9]: ./examples/blurred_example.jpg "Blurred Center Image"

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
* video.mp4 containing the video of the vehicle driving on the first track
* ericlavigne-data containing the tiny dataset that is referenced

Video of the vehicle driving from the simulator perspective can be found [here](https://www.youtube.com/watch?v=zdtLTEGDUAA)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a replica of the neural network used by Nvidia to train their self driving car.

The model includes RELU activation within the convolutional layers to introduce nonlinearity (model.py lines 77-81), and the data is normalized in the model using a Keras lambda layer (model.py line 75).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 83).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 92). The data was shuffled and 20% of the data was held back from training and instead used for validation. Shuffling ensured that the same 20% was not used every time.  Half of the dat used was also flipped to prevent the model from overfitting to the specific direction of the track.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and images from both the left and right camera. The training data set was from Eric Lavigne's "Tiny Dataset." All images were hand picked and steering angles adjusted manually.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to mimic the proven model that Nvidia used to create their own self-driving car.

My first step was to use a convolution neural network model similar to Nvidia's model. I thought this model might be appropriate because the developers at Nvidia used this model to successfully train a car to drive on the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a very low validation loss but failed to drive properly on the track. The vehicle could handle the beginning of the track very well, but lost control on the bridge, exiting the bridge, or on the turn after the bridge. I realized there was some sort of disconnect going on between the model inputs and the driving inputs. I also realized the testing of the model would take a while and upped the goal speed in drive.py from 9 to 12 miles per hour.

To combat the driving issue, I first attempted to remove extra data from other tracks and purposefully over-fit the model to the first track. However, the vehicle began driving erratically and quickly drove off the road.

Then I decided to try out a generator to see if I would have any luck with a smaller training set. However, this did not work either. After implementing the generator and adding back in the training data I had removed, the car drove down the track but over-corrected for every mistake which led to an increasing oscillation in the driving pattern. Eventually, the car kept getting stuck driving into the side of the bridge or veering off the track shortly before.

Clearly the generator was not working. Because I had so few images to train on, I decided to go back to augmenting the data manually and storing the set in memory. This would most likely have to be fleshed out with more augmentation and data in a larger scale vehicle in the real world, but for this project it worked well. However, after the two sharp turns after the bridge, the model always veered off the track sharply to the left.

Clearly I was still missing something. I went back and did more research on CV2, the Nvidia model, and the drive.py file. I found that cv2 was importing images in the BGR color space, drive.py was reading the inputs from the car in RGB, and the Nvidia model was designed for images in the YUV color space. I also noticed some discolorations in the surface of the road would confuse the car. To combat both of these issues, I converted all inputs to YUV (both in model.py and drive.py), and I applied a random gaussian blur to the training data to hopefully remove some of the confusing artifacts in the road. To quell some of the erratic driving, I lowered the compensation value from 0.2 to 0.1 for the left and right images. This way, the vehicle should not swerve across the track when it tries to correct itself.

I fired up the simulator one last time and it worked! The vehicle drives around the first track flawlessly. The driving is smooth, centered, and comfortable.  

Pushing my luck, I decided to run the car on the second track (the jungle), and immediately noticed a few issues. Often, the car would be confused about which track to take when the track doubled back past itself. Also, the vehicle lost its way whenever it encountered a shadow. The confusion about which track to take may be able to be addressed by cropping the sides of the images in a few pixels to limit the viewing field of the car. The shadow issue could probably be fixed by adding a random brightness augmentation to the training data. However, those will both require more research.

For now, the car drives flawlessly around the first track and with a few major flaws on the second.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74-88) consisted of a convolution neural network with the following layers and layer sizes:

* Input layer with lambda normalization
* Cropping layer to remove unnecessary background noise
* Convolutional layer 1 outputs 24 normalized planes with dimensions of 31x98 using a 5x5 kernel, subsample of 2x2 and a RELU activation
* Convolutional layer 2 outputs 36 normalized planes with dimensions of 14x47 using a 5x5 kernel, subsample of 2x2 and a RELU activation
* Convolutional layer 3 outputs 48 normalized planes with dimensions of 5x22 using a 5x5 kernel, subsample of 2x2 and a RELU activation
* Convolutional layer 4 outputs 64 normalized planes with dimensions of 3x20 using a 3x3 kernel and a RELU activation
* Convolutional layer 5 outputs 64 normalized planes with dimensions of 1x18 using a 3x3 kernel and a RELU activation
* Dropout layer (50% dropout)
* Flatten layer flattens to 1164 neurons
* Fully-connected (dense) layer outputs 100 neurons
* Fully-connected layer outputs 50 neurons
* Fully-connected layer outputs 10 neurons
* Fully-connected layer outputs 1 neuron (steering angle decision)


Here is a visualization of the architecture:

![Nvidia Model Diagram][image1]

#### 3. Creation of the Training Set & Training Process

The dataset used was Eric Lavigne's "Tiny Dataset" from slack.

To capture good driving behavior, the set contains mostly center lane driving. Here is an example image of center lane driving:

![alt text][image2]

The set also contains a few images of the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it veered too far to either side. These images show what a recovery looks like starting from the right and then the left :

![alt text][image3]
![alt text][image4]


This process is repeated on the jungle track and the mountain in order to get more data points.

![alt text][image5]
![alt text][image6]

To augment the data set, I flipped all images and steering angles to make the model more robust. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

This seems to have worked as the car is able to drive clockwise around the first track  as well as counter clockwise.

After the collection and augmentation processes, I had 642 total data points. I then preprocessed this data by converting the images to the YUV color space and applying a random gaussian blur. Here is an example of the blur:

![alt text][image2]
![alt text][image9]


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by a consistent increase in the validation loss after the third epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
