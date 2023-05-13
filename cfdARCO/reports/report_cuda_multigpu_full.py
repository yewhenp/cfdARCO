import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("report_cuda_multi_gpu_history.csv")

df_std = df.groupby(["mesh_sizes"]).std().reset_index()
df = df.groupby(["mesh_sizes"]).min().reset_index()


df["time_microseconds_cuda_single_gpu"] = df["time_microseconds_cuda_single_gpu"] / 1000000
df["time_microseconds_cuda_multi_gpu"] = df["time_microseconds_cuda_multi_gpu"] / 1000000
df["mesh_sizes"] = df["mesh_sizes"] ** 2
df_std["time_microseconds_cuda_single_gpu"] = df_std["time_microseconds_cuda_single_gpu"] / 1000000
df_std["time_microseconds_cuda_multi_gpu"] = df_std["time_microseconds_cuda_multi_gpu"] / 1000000
df_std["mesh_sizes"] = df_std["mesh_sizes"] ** 2

print(df)
print(df_std)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["time_microseconds_cuda_single_gpu"], label="One GPU")
plt.plot(df["mesh_sizes"], df["time_microseconds_cuda_multi_gpu"], label="Two GPUs")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Time, seconds')
plt.legend()
plt.savefig('cuda_multi_gpu_time.pdf')


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["time_microseconds_cuda_single_gpu"] / df["time_microseconds_cuda_multi_gpu"], label="Time 2 GPUs faster")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Times')
plt.legend()
plt.savefig('cuda_multi_gpu_speedup.pdf')

plt.show()
