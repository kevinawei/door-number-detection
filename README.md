# door-number-detection
Project 1 Room Number Detection

Kaiwen Wei

My model was trained using the SVHN(Street View House Numbers) dataset which consists of pictures that contain tens of thousands of pictures 
of training data as well as thousands of other pictures of digits that can be used for testing. 
The framework for the model was created by Jeong Joon Sup with me adding some minor modifications and changing the code so that it accepted videos
as input and outputted a text file with the results. 
The code for reading videos and processing the data was taken partially from Adrian Rosebrock’s Pyimagesearch text detection.
The model is a convolutional neural network consisting of 2 convolutional layers of dimension 32x32x32.
The output from those layers is then pooled using Max Pooling to a 16x16x32 output and then goes through 2 more convolutional layers before 
being pooled again to 8x8x64 dimensional outpu. It then has 2 final FC layers with an output of dimension 1x1x10 (because there are 10 possible digits and thus 10 classes).
There was also a dropout of .5 used in the first FC layers. The model uses relu activation functions and categorical cross entropy as its loss function.
I tried playing around with the activation functions and found that relu yielded the best results given the training set. 
Changing the loss function to mean squared error also yielded worse results so I decided to leave that as it was. 

====================================================================================================================
In order to run the code put the video into the working directory and type this command line: 

$ Detect_.py  --video Name of video file

There will then be a prompt for the name of the output txt file.

Afterward, when the detection is complete there will be a message that says “Detection Complete” at which point the file 
will have been created and will contain lines detailing the frame number as well as where and what digits are detected on that frame. 


Projects Cited:
SVHN Deep Digit Detector: Author – Jeong Joon Sup
https://github.com/penny4860/SVHN-deep-digit-detector

Pyimagesearch OpenCV Test Detection (EAST Text Detector): Author – Adrian Rosebrock
https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
