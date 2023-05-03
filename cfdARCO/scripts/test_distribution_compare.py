import argparse

import pandas as pd
import os
import subprocess


def run_one_run(mesh_path: str, available_nodes, procs_per_node):
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO"

    nodes_to_run = []
    i = 0
    for q in range(len(available_nodes)):
        one_node_list = []
        for _ in range(procs_per_node):
            one_node_list.append(available_nodes[i])
        nodes_to_run.append(",".join(one_node_list))
        i += 1

    command_ln = ["mpirun", "--oversubscribe", "--host", ",".join(nodes_to_run), bin_file, "-m", mesh_path, "-v", "--skip_history", "-d", "ln"]
    result_ln = subprocess.run(command_ln, capture_output=True, text=True)
    outs_ln = result_ln.stdout

    time_ln_str = outs_ln.split("\n")[-2].split(" ")[-1].split("[")[0]
    time_ln_microseconds = int(time_ln_str)

    command_cl = ["mpirun", "--oversubscribe", "--host", ",".join(nodes_to_run), bin_file, "-m", mesh_path, "-v", "--skip_history", "-d", "cl"]
    result_cl = subprocess.run(command_cl, capture_output=True, text=True)
    outs_cl = result_cl.stdout

    time_cl_str = outs_cl.split("\n")[-2].split(" ")[-1].split("[")[0]
    time_cl_microseconds = int(time_cl_str)

    print(f"Res(mesh={mesh_path}) = ln:{time_ln_microseconds}  cl:{time_cl_microseconds}")

    return time_ln_microseconds, time_cl_microseconds


def generate_report(mesh_pathes, available_nodes, procs_per_node, output_file):
    times_ln_microseconds = []
    times_cl_microseconds = []
    mesh_path_df = []
    for mesh_path in mesh_pathes:
        time_cur = run_one_run(mesh_path, available_nodes, procs_per_node)
        mesh_path_df.append(mesh_path)
        times_ln_microseconds.append(time_cur[0])
        times_cl_microseconds.append(time_cur[1])

    df_dict = {'mesh_path_df': mesh_path_df, 'times_ln_microseconds': times_ln_microseconds, "times_cl_microseconds": times_cl_microseconds}
    df = pd.DataFrame(df_dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHO bench')
    parser.add_argument('-o', '--out_file', required=False, default="report_distribution_compare.csv")
    parser.add_argument('-n', '--nodes', nargs='+', required=True, type=str)
    parser.add_argument('-p', '--pathes', nargs='+', required=True, type=str)
    parser.add_argument('--procs_per_node', required=True, type=int, default=4)

    args = parser.parse_args()

    generate_report(args.pathes, args.nodes, args.procs_per_node, args.out_file)

