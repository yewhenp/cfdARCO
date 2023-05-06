import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("report_per_nodes_full.csv", index_col=None)
df = df.drop(columns=["Unnamed: 0"])
df = df.groupby(['num_proc', 'mesh_sizes'])
df = df.min().reset_index()

df["times_microseconds"] = df["times_microseconds"] / 1000000
df["mesh_sizes"] = df["mesh_sizes"] ** 2

df = df.pivot(index='num_proc', columns='mesh_sizes', values='times_microseconds')
df = 1 / df.div(df.iloc[0])
df = df[:8]
print(df)
df.plot(legend=True)
plt.show()
