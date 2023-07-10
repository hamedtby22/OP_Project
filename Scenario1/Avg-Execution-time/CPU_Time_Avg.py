import numpy as np
import time

sizes = [100, 200, 500, 1000, 2000]   # اندازه‌های مختلف ماتریس‌ها

for size in sizes:
    a_cpu = np.random.rand(size, size)
    b_cpu = np.random.rand(size, size)

    times = []
    for i in range(10):   # اجرای تابع ۱۰ بار برای محاسبه میانگین زمان اجرا
        start_time = time.time()
        c_cpu = np.dot(a_cpu, b_cpu)
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"Matrix size: {size} x {size}, average execution time: {avg_time} seconds")