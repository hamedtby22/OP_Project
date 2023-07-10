import numpy as np
import matplotlib.pyplot as plt
import time

sizes = [100, 200, 500, 1000, 2000]   # اندازه‌های مختلف ماتریس‌ها

times = []
for size in sizes:
    a_cpu = np.random.rand(size, size)
    b_cpu = np.random.rand(size, size)

    start_time = time.time()
    c_cpu = np.dot(a_cpu, b_cpu)
    end_time = time.time()

    times.append(end_time - start_time)

plt.plot(sizes, times, marker='o')
plt.title('CPU Execution Time for Matrix Multiplication')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (seconds)')
plt.show()