import numpy as np
import cupy as cp
import time

sizes = [100, 200, 500, 1000, 2000]   # اندازه‌های مختلف ماتریس‌ها

for size in sizes:
    a_cpu = np.random.rand(size, size)
    b_cpu = np.random.rand(size, size)

    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)

    cpu_times = []
    gpu_times = []
    for i in range(10):   # اجرای تابع ۱۰ بار برای محاسبه میانگین زمان اجرا
        start_time = time.time()
        c_cpu = np.dot(a_cpu, b_cpu)
        end_time = time.time()

        cpu_times.append(end_time - start_time)

        start_time = time.time()
        c_gpu = cp.matmul(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()   # همگام‌سازی برای انتظار پایان اجرای تابع
        end_time = time.time()

        gpu_times.append(end_time - start_time)

    cpu_avg_time = sum(cpu_times) / len(cpu_times)
    gpu_avg_time = sum(gpu_times) / len(gpu_times)
    print(f"Matrix size: {size} x {size}, CPU average execution time: {cpu_avg_time} seconds, GPU average execution time: {gpu_avg_time} seconds")