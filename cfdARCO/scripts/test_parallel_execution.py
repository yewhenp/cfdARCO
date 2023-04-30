import argparse

import pandas as pd
import os
import subprocess


def run_one_run(num_proc: int, available_nodes=("node1", "node2", "node3")):
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO"

    nodes_to_run = []
    i = 0
    for q in range(num_proc):
        nodes_to_run.append(available_nodes[i])
        i += 1
        if i >= len(available_nodes):
            i = 0

    command = ["mpirun", "--host", ",".join(nodes_to_run), bin_file]
    result = subprocess.run(command, capture_output=True, text=True)
    outs = result.stdout

    time_str = outs.split("\n")[-2].split(" ")[-1].split("[")[0]
    time_microseconds = int(time_str)

    print(f"Procs({num_proc}) = {time_microseconds}")

    return time_microseconds


def generate_report(max_num_proc: int, available_nodes=("node1", "node2", "node3"), output_file="report.csv"):
    num_proc = []
    times_microseconds = []
    for i in range(1, max_num_proc+1):
        time_cur = run_one_run(i, available_nodes)
        num_proc.append(i)
        times_microseconds.append(time_cur)

    dict = {'num_proc': num_proc, 'times_microseconds': times_microseconds}
    df = pd.DataFrame(dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHO bench')
    parser.add_argument('-p', '--num_proc', required=True, type=int)
    parser.add_argument('-o', '--out_file', required=False, default="report.csv")
    parser.add_argument('-n', '--nodes', nargs='+', required=True, type=str)

    args = parser.parse_args()
    generate_report(args.num_proc, args.nodes, args.out_file)

