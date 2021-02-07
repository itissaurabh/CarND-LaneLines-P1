import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from test_helper import *


# Using Grayscale images to build pipeline

image_list = []

def pipeline(image):
    """Process a colored image to detect the lane markings

    We do the following steps:
    1. Convert the image to grayscale
    2. Adjust the contrast
    3. Blur the image
    4. Detect edges using Canny algorithm
    5. Mark the image to ignore un-necessary area
    6. Detect lines using Hough Transform
    7. Extend the detected lines
    """

    img = grayscale(image)
    blurred_img = gaussian_blur(img, 5)
    cimg = canny(blurred_img, 50, 150)

    rows = image.shape[0]
    cols = image.shape[1]
    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.4, rows * 0.6]
    right_top = [cols * 0.6, rows * 0.6]

    breakpoint()
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    print(rows, cols, vertices)
    masked_image = region_of_interest(cimg, vertices)

    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40  # minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)

    annotated_image = weighted_img(line_img, image, α=0.7, β=1., γ=0.)

    display_imgs([img, cimg, masked_image, annotated_image])

    return annotated_image


def main():
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    yellow_output = 'test_videos_output/solidYellowLeft.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
    clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)

    challenge_output = 'test_videos_output/challenge.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
    clip3 = VideoFileClip('test_videos/challenge.mp4')
    challenge_clip = clip3.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)

#    for file in os.listdir("test_images/"):
#        annotated_image = process_image(mpimg.imread(f"test_images/{file}"))


if __name__ == "__main__":
    main()

