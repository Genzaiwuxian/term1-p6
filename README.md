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

Here is an example using the `gray` color space (the classifier use color space "0") and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/car_HOG.png)
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/not_car_HOG.png)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, expecially the color space, i found the 'RBG' will do not work well, many non-cars image will be labeled; HLS have a good performance, however the further black car doesn't work well, often miss it. Finally, The 'LUV' can find most of cars and work well, so i chose this color space.
HOG parameters:
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"

and there are also some parameters i choose:
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

#### 3. trained a classifier.

Before train, is normalized the train data by using :(cell 53)
- X_scaler = StandardScaler().fit(X) # Fit a per-column scaler
- scaled_X = X_scaler.transform(X) # Apply the scaler to X

I trained a linear SVM using loss='hinge'.
the input are combination of HOG, color space (LUV), histogram.. the code is in cell 53.
some results are
- Car samples:  10292
- Notcar samples:  10136
- Using: 8 orientations 8 pixels per cell and 2 cells per block
- Feature vector length: 2432
- 3.1 Seconds to train SVC...
- Test Accuracy of SVC =  0.9949

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

- The code are in cell 30/50/54
- i choose the ROI is x_start_stop=[None, None], y_start_stop=[400, 640], mostly is the low half image, for most upper half image information is ueslless, most are sky and trees.
- i tried some size of slding windows, some results are:
- Window_size= 90 and 160
- the overlap is 0.85
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/test_classifier.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

- the windows is too much, so put into new ROI, two sides to detect new coming nearby vehicles, and area to detected front away vehcles
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/ROI.png)
- the slding window is also can not fit the vehicle size, the ideas are: find the center of vehicle, and max points of rectangle, then make some scale of the lengh and width, and re-plot a rectangle based on center and new width & length
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/image_output.png)

### Video Implementation

Here's a link to video: https://github.com/Genzaiwuxian/term1-p6/blob/master/output_video.mp4


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

- some false positives, like road, side infrustrucure. etc, most using the heat map and set threshold (3 in my code) to filer the false detection.
- the overlap counding boxes use the idea with 'fit vehicle shapre size, it calculate the center of vehicle based on maximum rectangle of detected windows, and then calculate width and length, conver it into new sacle length/width, finally re-plot the new rectangle.

Here's an example result showing the heatmap:
![image](https://github.com/Genzaiwuxian/term1-p6/blob/master/output_images/heat_map.png)


### Discussion

- some vehicle final detected windows also can not reflect the real shape, that's may influence the vehicle next movement. so the code for re-shape can be smooth in further
- last project is advanced lane detection, and it can join together with this project show in same video.

