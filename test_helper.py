import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#----------------------------------------------------

def draw_lines_orig(img, lines, color=[255, 0, 0], thickness=2):
    """
    Separate line segments by their slope ((y2-y1)/(x2-x1)).
    Use the slope-intercept form to extrapolate the line.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines_orig(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn alongwith the lines array.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_orig(line_img, lines)
    return line_img


#----------------------------------------------------

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Separate line segments by their slope ((y2-y1)/(x2-x1)).
    Use the slope-intercept form to extrapolate the line.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).

    It may happen that only left-lane or right-lane is drawn depending on
    detection.
    """

    left_lane, right_lane = seggregate_and_average_lane_lines(lines)

    extrapolated_lines = []

    if left_lane.size != 1: # Make sure left_lane is not 'nan'
        left_line = extrapolate_lines(img.shape, left_lane)
        extrapolated_lines.append([left_line])

    if right_lane.size != 1:  # Make sure left_lane is not 'nan'
        right_line = extrapolate_lines(img.shape, right_lane)
        extrapolated_lines.append([right_line])

    for line in extrapolated_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn alongwith the lines array.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(line_img, lines, thickness=6)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def seggregate_and_average_lane_lines(lines):
    """
    Separate line segments by their slope ((y2-y1)/(x2-x1)).
    Slope < 0 => left-lane
    Slope > 0 => left-lane

    Return the Average for left and right lane
    """
    left_lane = []
    right_lane = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            # Fit the coordinates into degree 1 polynomial for getting slope and intercept
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

            # Only lines within a certain slope qualify as lane-lines
            if slope < -0.5 and slope > -0.9:
                left_lane.append((slope, intercept))
            if slope < 0.9 and slope > 0.5:
                right_lane.append((slope, intercept))

    return np.average(left_lane, axis = 0), np.average(right_lane, axis = 0)


def extrapolate_lines(image_shape, line):
    """Use slope-intercept form for extrapolating the line.

    We draw from bottom of the image to 3/5th of the image height (the same
    height we chose for region of interest).

    Output of this function is the set of two coordinate points indicating the
    end-points of the extrapolated-line to be drawn.

    """
    slope, intercept = line
    y1 = image_shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])



def display_imgs(img_list, labels=[], cols=2, fig_size=(15,15)):
    """
    Helper function to display images in a grid form
    """
    if len(labels) > 0:
        # If label is provided, it must be provided for all images
        assert(len(img_list) == len(labels))

    # At lieast one image must be provided
    assert(len(img_list) > 0)

    cmap = None # All single dimenson images must be displayed in 'gray'
    rows = math.ceil(len(img_list) / cols)

    plt.figure(figsize=fig_size)

    for i in range(len(img_list)):
        plt.subplot(rows, cols, i+1)

        if len(img_list[i].shape) == 2:
            cmap = 'gray'

        if len(labels) > 0:
            plt.title(labels[i])

        plt.imshow(img_list[i], cmap=cmap)

    plt.tight_layout()
    plt.show()


def to_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def process_image(image):
    """Process a colored image to detect the lane markings

    Input: Color (RGB) Image
    Output: Input image overlaid with lane markings
    """

    # Step 1: Image Pre-processing to highlight lane markings

    # Convert to grayscale
    gsimg = grayscale(image)

    # Convert to HLS, isolate Yellow and White Color
    hlsimg = to_hls(image)
    white_mask = cv2.inRange(hlsimg,
                             np.array([0, 200, 0], dtype=np.uint8),
                             np.array([200, 255, 255], dtype=np.uint8))
    yellow_mask = cv2.inRange(hlsimg,
                              np.array([10, 0, 100], dtype=np.uint8),
                              np.array([40, 255, 255], dtype=np.uint8))
    yw_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Use the yellow-white mask to highlight the lane markings
    yw_masked_image = cv2.bitwise_and(gsimg, gsimg, mask=yw_mask)

    # Remove the noise in the image with a 5x5 Gaussian filter
    blurred_img = gaussian_blur(yw_masked_image, 5)

    # Step 2: Apply Canny Edge Detection
    cimg = canny(blurred_img, 50, 150)

    # Mask everything except the region of interest
    # As the markings are always in front, we only need to look at specific area in image.
    rows = image.shape[0]
    cols = image.shape[1]
    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.4, rows * 0.6]
    right_top = [cols * 0.6, rows * 0.6]

    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    roi_image = region_of_interest(cimg, vertices)

    # Step 3: Apply Hough Transform to extract line coordinates, and
    # extrapolate them to draw lane markings

    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20  # minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    line_img = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)

    # Merge the original image with the lane-marking image
    annotated_image = weighted_img(line_img, image, α=0.7, β=1., γ=0.)

    return annotated_image
