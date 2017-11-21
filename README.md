**Vehicle Detection Project**

The goals / steps of this project are the following:

* Analysis of data - Car and Non-car images
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier (loss='hinge')
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
Please refer to 'P6.ipynb' for details code. 

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Analysis of data - car and non car images
These images have to be extracted from real world videos and images, and correctly labeled. 
I use the images which provide by Udacity. The images have 64 x 64 pixels.

![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/car_image.png)
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/not_car_image.png)

### Histogram of Oriented Gradients (HOG)

#### 1. extracted HOG features from the training images.

The code for this step is contained in 3~5 code cell of 'P3.ipynb'.  

I started by reading in all the `vehicle` and `non-vehicle` images. 
Then randon selected one file number (less than 5000, cause car and not car images are more than 5000 filenames)
Here is an example of one of each of the `vehicle` and `non-vehicle` classes (same with data analysis):

![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/car_image.png)
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/not_car_image.png)

I then explored color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/car_HOG.png)
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/not_car_HOG.png)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, expecially the color space, i found the 'RBG' will do not work well, many non-cars image will be labeled; HLS have a good performance, however the further black car doesn't work well, often miss it. Finally, The 'LUV' can find most of cars and work well, so i chose this color space.

and there are also some parameters i choose:
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using loss='hinge'.
the input are combination of HOG, color space (LUV), histogram.. the code is in cell 53.
some results are
  Car samples:  10292
  Notcar samples:  10136
  Using: 8 orientations 8 pixels per cell and 2 cells per block
  Feature vector length: 2432
  3.1 Seconds to train SVC...
  Test Accuracy of SVC =  0.9949

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

