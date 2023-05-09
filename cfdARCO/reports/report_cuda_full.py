import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("report_cuda_full.csv")

df_std = df.groupby(["mesh_sizes"]).std().reset_index()
df = df.groupby(["mesh_sizes"]).min().reset_index()


df["times_microseconds_parallel"] = df["times_microseconds_parallel"] / 1000000
df["times_microseconds_cuda"] = df["times_microseconds_cuda"] / 1000000
df["times_microseconds_cuda_memcopy"] = df["times_microseconds_cuda_memcopy"] / 1000000
df["mesh_sizes"] = df["mesh_sizes"] ** 2
df_std["times_microseconds_parallel"] = df_std["times_microseconds_parallel"] / 1000000
df_std["times_microseconds_cuda"] = df_std["times_microseconds_cuda"] / 1000000
df_std["times_microseconds_cuda_memcopy"] = df_std["times_microseconds_cuda_memcopy"] / 1000000
df_std["mesh_sizes"] = df_std["mesh_sizes"] ** 2

print(df)
print(df_std)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["times_microseconds_parallel"], label="CPU, parallel")
plt.plot(df["mesh_sizes"], df["times_microseconds_cuda"], label="CUDA")
plt.plot(df["mesh_sizes"], df["times_microseconds_cuda_memcopy"], label="CUDA with memory moves")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Time, seconds')
plt.legend()


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["times_microseconds_parallel"] / df["times_microseconds_cuda"], label="Time CUDA faster")
plt.plot(df["mesh_sizes"], df["times_microseconds_parallel"] / df["times_microseconds_cuda_memcopy"], label="Time CUDA with memcopy faster")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Times')
plt.legend()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df_std["mesh_sizes"], df_std["times_microseconds_cuda"], label="CUDA, STD")
plt.plot(df_std["mesh_sizes"], df_std["times_microseconds_cuda_memcopy"], label="CUDA with memcopy, STD")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Time, seconds')
plt.legend()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["times_microseconds_cuda_memcopy"] / df["times_microseconds_cuda"], label="CUDA, STD")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Time, seconds')
plt.legend()

plt.show()
