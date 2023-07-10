import cupy as cp
import time

sizes = [100, 200, 500, 1000, 2000]   # اندازه‌های مختلف ماتریس‌ها

for size in sizes:
    a_gpu = cp.random.rand(size, size)
    b_gpu = cp.random.rand(size, size)

    times = []
    for i in range(10):   # اجرای تابع ۱۰ بار برای محاسبه میانگین زمان اجرا
        start_time = time.time()
        c_gpu = cp.matmul(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()   # همگام‌سازی برای انتظار پایان اجرای تابع
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"Matrix size: {size} x {size}, average execution time: {avg_time} seconds")