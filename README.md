# Behavioral Cloning

## Overview
The goal of this project was to teach a car to drive autonomously in a simulator by using recorded image sequences along with steering angle labels as input and train a convolutional neural network to output correct steering angles based on new streaming camera images.

## Collecting Data
| Left          | Center        | Right  |
| ------------- |:-------------:| ------|
|![Left] (https://github.com/jimwinquist/behavioral-cloning/blob/master/images/left_raw.jpg) | ![Center] (https://github.com/jimwinquist/behavioral-cloning/blob/master/images/center_raw.jpg) | ![Right] (https://github.com/jimwinquist/behavioral-cloning/blob/master/images/right_raw.jpg)

To collect training data I began by driving several smooth laps around the track in the simulator at a consistent speed around 30mph. I drove 3 laps of the training track in one direction and then drove 3 laps of the track in the reverse direction, trying to keep in the center of the lane as much as possible. The car in the simulator is mounted with 3 forward facing cameras on the center left and right of the car and I used the images from all three to train with. The initial training data set consisted of approximately 40,000 images including the center, left, and right camera images for each frame.

After training and testing in the simulator, this initial data set wasn't enough data to accurately predict certain sharp turns, so I found it necessary to collect additional training examples of spots that the model had trouble predicting. I recorded additional data in the problem sections of the track by slowing the car down to approximately 5mph to ensure that I would gather a larger number of images in these sections to include in the training set. This was an iterative process that required gathering data, training the model, testing in the simulator and then collecting more data to augment the training set.

## Processing Images
<img src="https://github.com/jimwinquist/behavioral-cloning/blob/master/images/processed.png" width="160">

To process the raw images from the camera I began by converting the color space from RGB to YUV with the idea that the YUV color space would generalize better to changing lighting conditions or changes in the darkness/lightness of road surfaces. After converting the color space, I found that training with the full images took a considerable amount of time and was causing the model to be less accurate so I scaled the images down by 75%. The raw images from the camera had height: 160, width:320, and 3 color channels(160x320x3) After resizing I ended up with images that were 40x80x3. This made the model train about 5 times faster and reduced the number of features that the model needed to train on which improved the accuracy of the predictions. In addition to resizing, I also noticed that cropping the image to reduce the amount of extraneous information in the image such as trees and sky also helped improve performance of the model. I cropped off the top 25% of the image so the model only needed to focus on the road surface and cropped off the bottom 10% which was filled mostly by the hood of the car.

## Model Architecture
I had read Nvidia's [End to End Learning for Self-Driving Cars] (https://arxiv.org/abs/1604.07316) and commaai's [Learning a Driving Simulator] (https://arxiv.org/abs/1608.01230) and I began by experimenting with both of these models. After testing both I settled on using the commaai model because it was very similar to one I had used previously for doing traffic sign classification and I was familiar with it. Also, on initial testing I found the commaai model to perform better and it was easier to modify. I could also see consistent progress indicating that the model was learning with this approach. In addition to the base model I added a Normalization Layer as the first layer to the model to normalize the input images, and I added ELU activations and Batchnormalization and Dropout layers to zero-mean the weights at each layer and reduce overfitting respectively. Here is an overview of my model:

<img src="https://github.com/jimwinquist/behavioral-cloning/blob/master/images/model.png" width="120">

### Behavioral Cloning Model
- **Normalization** input = (25, 80, 3)
- **2D Convolution** 16 Filters with a 5x5 kernel, 4x4 stride, and SAME padding. output(7x20x16)
- **BatchNormalization**
- **ELU**
- **2D Convolution** 32 Filters with a 5x5 kernel, 2x2 stride, and SAME padding. output(4x10x32)
- **BatchNormalization**
- **ELU**
- **2D Convolution** 64 Filters with a 5x5 kernel, 2x2 stride, and SAME padding. output(2x5x64)
- **BatchNormalization**
- **ELU**
- **Flatten** output (640)
- **Dropout(0.2)**
- **ELU**
- **Fully Connected** output (512)
- **BatchNormalization**
- **Dropout(0.5)**
- **ELU**
- **Output**


## Training the Model
As my training set grew to approximately 50,000 images it became increasingly harder to hold the data in memory so I made use of keras fit generator to pass batches of images to the model for training. For each frame of driving I also augmented the collected data, by inferring additional images and their angle offsets. I passed the center image from the car and it's steering angle directly to the model. I also flipped the center images and negated the steering angle to simulate the car turning the opposite direction on the road. I also made use of the left and right camera images by adding an angle offset of -0.25 for the right camera image and +0.25 for the left camera image. By doing this I was able to help the model simulate what it should do if the car began to drift away from the center of the road, effectively teaching how to recover when drifting towards the edge of the lanes. I initially tested training for 10 or more epochs but found that the model had generally converged after only about 5 epochs. So, I tested the base training set for 5 epochs, and then began testing it's autonomous capability in the simulator. Based on past experience with Gradient Descent not converging without a lot of parameter tuning, I decided to use an Adam Optimizer with initial learning rate of 0.0001. On the first pass the model was able to complete about 50% of the track. So I went back and gathered more data for some of the problem areas and retrained the model with this new data and a lower learning rate of 0.000001. After 3 iterations of gathering more data for trouble areas, I was able to get a model which successfully completed the whole track.

## Ideas for Future Exploration
While I was able to successfully train a model to drive autonomously using only front facing camera images, I believe a lot more work would need to be done to test and make sure the model could generalize well to a much larger variety of road conditions. In the future I would like to experiment more with using additional color spaces or image processing using either HSL or Canny images, to see if the model could generalize to a wider range of lighting conditions, and colors of road surface. I would also like to experiment with additional depths of model architecture to find ways to improve the learning capability of the network. I think to really be sure of the models performance it would be necessary to collect data from a much larger range of driving situations and road conditions.

Overall this was a really exciting project and it taught me a lot about the power of deep learning to solve interesting regression problems such as predicting steering angles. I also came away with a strong understanding of how necessary it is to generate augmented data to supplement and strengthen the distribution of data to infer plausible but unseen situations. It is also such a thrill to see a model you have trained driving a car autonomously for the first time. Feels like magic!