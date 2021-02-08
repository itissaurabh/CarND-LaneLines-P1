# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./writeup_images/pipeline_summary.png "Pipeline Summary"
[image2]: ./writeup_images/gs_failure.png "Scenario for GrayScale Failure"
[image3]: ./writeup_images/why_hls_masking.png "HLS Masking"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The following image summarises the pipeline and we will refer to it in description below.

![Image Representing Pipeline Summary][image1]

The pipeline broadly relies on 

1. Pre-processing the image to highlight the lane markings and isolate the
   section containing lane markings.
   - Convert the image to Grayscale
   - Convert the image to HLS and isolate yellow and white regions and create a
     yellow-white region mask.
   - Use the yellow-white mask to highlight lane markings
   - Gaussian blur the image using a 5x5 kernel
2. Applying Canny Edge Detection Algorithm to detect the edges of Lane markings.
   - Apply Canny Edge Detector b. Create a trapezoidal region of interest to
     isolate the lane markings in front. As the car camera is always going to
     look in forward direction, lane markings are always going to be located in
     a specific region of image. Thus, we can ignore rest of the rest of the
     image.
3. Get line coordinates of these Lane Markings using Hough Transform.
   - Use Hough Transform to detect lines.
   - Convert the lines into slope-intercept form.
   - Based on the orientation of camera, the slope of lane lines will be within
     a specific range. Segregate the lines into those that will part of left
     lane and right lane based on the slope. Discard all lines with slope that
     does not seem to fall in either category (example, horizontal line)
   - Average the lines to get an approximation of the left and right lane marking.
   - Extract the line coordinates using the slope-intercept value of left and
     right lane. The line should go from bottom of the image to roughly the top
     of region of interest (i.e. 3/5th of the image height)
   - Draw a line using the extracted coordinates.
   - Merge the original Image with the image of lines drawn.
   - Note: The pipeline presently chooses to skip drawing lines in case we are
     not able to detect lane markings. This is by design as we do not want
     incorrect estimate of lane lines if detection fails.

##### Note:
- While grayscale images are good for most of the cases, specific cases like the
  cemented road in between the black tarmac (challenge video), causes challenges
  in proper detection due to the lack of contrast. As shown in figure below, in
  grayscale you can barely identify the yellow lane marking.

![Image indicating Scenario for GrayScale Failure][image2]

- This results in the lane line not getting detected properly by Canny Edge
  detector and hence the pipeline breaks when trying to estimate lane lines.
- In order to handle these type of scenarios, we have to rely on the color of
  lane lines. In HLS space, we can use the Hue, Lightness and Saturation
  information to better detect the lane lines.
  - For detecting yellow, we filter on a specific Hue channel, discarding low
    saturation yellow's to skip the yellow of grass and hills. For white
    detection, we rely of lightness values in the high range.
  - We create a mask by or-ing the white filtered and yellow filtered image and
    use this mask on the grayscale image to create an image with high contrast
    lane markings. 

![Image indicating steps for HLS based masking][image3]

### 2. Identify potential shortcomings with your current pipeline


- The HLS color selection is very much dependent on the color filtering that was
  optimal on the test data provided. It is very much possible that if the road
  marking color changes a bit because of say dust or snow or some other
  obstruction, the lane marking is undetectable.
  
- Region of Interest is hardcoded based on the image types that were provided as
  test cases. In case of, say a more curved or a steep road, the region of
  interest selection can prove to be sub-optimal.
  
- Parameters for Canny and Hough are selected based on multiple rounds of trial
  and error on the three test videos provided. There should probably be a better
  way to do the same, which is more suitable for varying conditions. Presently
  it does not give a very good confidence that the pipeline will work in all
  scenarios.

- The angle of the lane markings used for filtering is again fine-tuned based on
  experimentation on the provided test data. If because of say a hilly terrain,
  the lane marking slope changes quite a bit, this logic would fail.
  
- What would happen when there is a steep turn on the road. The Hough detection
  maps the road to a linear segment as of now. The curve fitting used in
  draw_lines considers the lane markings as straight lines. If we can use a
  higher order curve, it would map to the curves much better.
  
- If for some reason, lane marking is not detected on the road (color change,
  shading or some other unexpected scenario), the entire frame is missed. As we
  already may have some data from previous detection, we can work out some way
  extrapolate from there and have partial/indicative lane markings. For this
  extrapolation, we will need speed parameter also to determine how far the car
  has travelled since last detection and probably angular shift data (maybe from
  Inertial Navigation System). This is essential as we do not want to do
  incorrect detection, which can be disastrous for the vehicle..

### 3. Suggest possible improvements to your pipeline

- Use higher order curve fitting to detect curves.
- Make the lane detection logic color independent and region of interest
  independent by using CNN or some other appropriate technique.
- Consider the position of the lane lines in previous images in order to
  validate and influence the calculation of the current results. We would need
  data from other sensors for this to be safe.
- Overall, the pipeline in its current state does not inspire much confidence
  due to the experimental fine-tuning of parameters. The lane marking data needs
  to combined with data from other sensors to fill up the gaps if they occur
  during detection.
