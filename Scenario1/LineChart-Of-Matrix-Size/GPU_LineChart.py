import cupy as cp
import matplotlib.pyplot as plt
import time

sizes = [100, 200, 500, 1000, 2000]   # اندازه‌های مختلف ماتریس‌ها

times = []
for size in sizes:
    a_gpu = cp.random.rand(size, size)
    b_gpu = cp.random.rand(size, size)

    start_time = time.time()
    c_gpu = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()   # همگام‌سازی برای انتظار پایان اجرای تابع
    end_time = time.time()

    times.append(end_time - start_time)

plt.plot(sizes, times, marker='o')
plt.title('GPU Execution Time for Matrix Multiplication')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (seconds)')
plt.show()