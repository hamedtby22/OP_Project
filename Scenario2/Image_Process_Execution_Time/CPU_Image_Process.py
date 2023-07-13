import time
import cv2
from skimage import io

# یک تصویر را برای پردازش بارگیری می کنیم
image_path = "https://iiif.lib.ncsu.edu/iiif/0016007/full/800,/0/default.jpg"
image = io.imread(image_path)

#تابع نمایش تصویر
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# پردازش تصویر سی پی یو
def cpu_image_processing(image):
    start_time = time.time()

# انجام عملیات پردازش تصویر با استفاده از سی پی یو
#تبدیل تصویر به مقیاس خاکستری
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    cv_show('My Image', gray_image)

    end_time = time.time()
    execution_time = end_time - start_time
    return gray_image, execution_time


# پردازش تصویر سی پی یو را اجرا می کنیم
cpu_result, cpu_execution_time = cpu_image_processing(image)

print("CPU Execution Time:", cpu_execution_time, "seconds")
