import argparse

import pandas as pd
import os
import subprocess


def run_one_run(mesh_size: int):
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO"

    # command_cuda = ["mpirun", "-n", "2", bin_file, "--skip_history", "-L", str(mesh_size), "-d", "ln", "-t", "300", "-c"]
    command_cuda = [bin_file, "--skip_history", "-L", str(mesh_size), "-d", "ln", "-t", "300", "-c"]
    result_cuda = subprocess.run(command_cuda, capture_output=True, text=True)
    outs_cuda = result_cuda.stdout

    time_str_cuda = outs_cuda.split("\n")[-2].split(" ")[-1].split("[")[0]
    time_microseconds_cuda = int(time_str_cuda)

    # command_parallel = ["mpirun", "-n", "8", bin_file, "--skip_history", "-L", str(mesh_size), "-d", "ln", "-t", "300"]
    # result_parallel = subprocess.run(command_parallel, capture_output=True, text=True)
    # outs_parallel = result_parallel.stdout
    #
    # time_str_parallel = outs_parallel.split("\n")[-2].split(" ")[-1].split("[")[0]
    # time_microseconds_parallel = int(time_str_parallel)
    time_microseconds_parallel = 0

    print(f"Res(mesh_size={mesh_size}) = cuda - {time_microseconds_cuda}  parallel - {time_microseconds_parallel}")

    return time_microseconds_cuda, time_microseconds_parallel


def generate_report(mesh_sizes, output_file="report_cuda.csv"):
    times_microseconds_cuda = []
    times_microseconds_parallel = []
    mesh_sizes_df = []
    for mesh_size in mesh_sizes:
        time_cur_cuda, time_cur_parallel = run_one_run(mesh_size)
        mesh_sizes_df.append(mesh_size)
        times_microseconds_cuda.append(time_cur_cuda)
        times_microseconds_parallel.append(time_cur_parallel)

    df_dict = {"mesh_sizes": mesh_sizes_df, "times_microseconds_cuda": times_microseconds_cuda,
               "times_microseconds_parallel": times_microseconds_parallel}
    df = pd.DataFrame(df_dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHO bench')
    parser.add_argument('-mf', '--mesh_size_from', required=True, type=int)
    parser.add_argument('-mt', '--mesh_size_to', required=True, type=int)
    parser.add_argument('-ms', '--mesh_size_step', required=True, type=int)
    parser.add_argument('-o', '--out_file', required=False, default="report.csv")

    args = parser.parse_args()

    mesh_sizes = range(args.mesh_size_from, args.mesh_size_to, args.mesh_size_step)
    generate_report(list(mesh_sizes), args.out_file)

