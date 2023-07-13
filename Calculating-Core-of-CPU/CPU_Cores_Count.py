import psutil
import csv

with open('cores-of-cpu.csv', mode='w', newline='') as core:
    writer = csv.writer(core)

    # سر تیتر جدول
    writer.writerow(['system-resources'])

    psutil.cpu_percent(interval=1)
    per_cpu = psutil.cpu_percent(percpu=True)
    # For individual core usage with blocking, psutil.cpu_percent(interval=1, percpu=True)
    for idx, usage in enumerate(per_cpu):
        writer.writerow([f"CORE_{idx+1}: {usage}%"])