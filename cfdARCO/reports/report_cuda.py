import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("report_cuda_2_proc.csv")

df["times_microseconds_parallel"] = df["times_microseconds_parallel"] / 1000000
df["times_microseconds_cuda"] = df["times_microseconds_cuda"] / 1000000
df["mesh_sizes"] = df["mesh_sizes"] ** 2

print(df)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["times_microseconds_parallel"], label="CPU, parallel")
plt.plot(df["mesh_sizes"], df["times_microseconds_cuda"], label="CUDA")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Time, seconds')
plt.legend()


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["times_microseconds_parallel"] / df["times_microseconds_cuda"], label="Time CUDA faster")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Timess')
plt.legend()

plt.show()
