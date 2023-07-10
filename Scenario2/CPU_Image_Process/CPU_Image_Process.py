import time
import cv2 as cv 
from google.colab.patches import cv2_imshow # for image display
from skimage import io

# Load an image for processing
image_path = "https://iiif.lib.ncsu.edu/iiif/0016007/full/300,/0/default.jpg"
image = io.imread(image_path)

# CPU Image Processing
def cpu_image_processing(image):
    start_time = time.time()

    # Perform image processing operations using CPU
    # Example: Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv2_imshow(gray_image)
    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time


# Run CPU Image Processing
cpu_result, cpu_execution_time = cpu_image_processing(image)

cv.waitKey(0)

print("CPU Execution Time:", cpu_execution_time, "seconds")