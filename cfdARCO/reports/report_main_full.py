import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("report_per_nodes_full.csv", index_col=None)
df = df.drop(columns=["Unnamed: 0"])
df["mesh_sizes"] = df["mesh_sizes"] ** 2
df["times_microseconds"] = df["times_microseconds"] / 1000000
df_group = df.groupby(['num_proc', 'mesh_sizes'])


df = df_group.min().reset_index()

df = df.pivot(index='num_proc', columns='mesh_sizes', values='times_microseconds')
df = 1 / df.div(df.iloc[0])
print(df)
df.plot(legend=True)


df = df_group.std().reset_index()

df = df.pivot(index='num_proc', columns='mesh_sizes', values='times_microseconds')
print(df)
df.plot(legend=True)


plt.show()
