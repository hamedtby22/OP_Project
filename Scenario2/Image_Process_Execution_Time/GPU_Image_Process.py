import cv2
import tensorflow as tf
import time
from skimage import io
from google.colab.patches import cv2_imshow # برای نمایش تصویر

# تصویر را بارگذاری می کنیم
img = io.imread('https://iiif.lib.ncsu.edu/iiif/0016007/full/300,/0/default.jpg')

# تبدیل تصویر به یک تانسور
img_tensor = tf.convert_to_tensor(img)

# تانسور را به حافظه جی پی یو منتقل کنید
with tf.device('/gpu:0'):
    img_tensor_gpu = tf.identity(img_tensor)


start_time = time.time()

# پردازش تصویر را روی جی پی یو انجام می دهیم
with tf.device('/gpu:0'):
    gray_tensor_gpu = tf.image.rgb_to_grayscale(img_tensor_gpu)
    thresholded_tensor_gpu = tf.where(gray_tensor_gpu > 127, 255, 0)

end_time = time.time()

#زمان اجرا را محاسبه می کنیم
execution_time = end_time - start_time

# تانسور آستانه ای را به حافظه سی پی یو بر می گردانیم
thresholded_tensor = thresholded_tensor_gpu.numpy()

# نمایش تصاویر اصلی و پردازش شده
cv2_imshow(img)
cv2_imshow(thresholded_tensor)
cv2.waitKey(0)
cv2.destroyAllWindows()

# زمان اجرا را چاپ کنید
print('execution_time: {:.4f} seconds'.format(execution_time))

#شیئی جبری است که رابطه چندخطی بین مجموعه‌ها و اشیاء جبری مربوط به یک فضای برداری را توصیف می‌نماید