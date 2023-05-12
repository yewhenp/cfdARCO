import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("report_per_nodes_full.csv", index_col=None)
df = df.drop(columns=["Unnamed: 0"])
df["Mesh sizes"] = df["mesh_sizes"] ** 2
df["Number of nodes"] = df["num_proc"]
df["times_microseconds"] = df["times_microseconds"] / 1000000

df.drop(columns=["num_proc", "mesh_sizes"])

df_group = df.groupby(['Number of nodes', 'Mesh sizes'])


df = df_group.min().reset_index()

df = df.pivot(index='Number of nodes', columns='Mesh sizes', values='times_microseconds')
df = 1 / df.div(df.iloc[0])
print(df)
df.plot(legend=True)
plt.savefig('8_aws_machines_experiment.pdf')

df = df_group.std().reset_index()

df = df.pivot(index='Number of nodes', columns='Mesh sizes', values='times_microseconds')
print(df)
df.plot(legend=True)
plt.savefig('8_aws_machines_experiment_std.pdf')

plt.show()
