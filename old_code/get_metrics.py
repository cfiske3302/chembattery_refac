import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)

    args = parser.parse_args()

    data = np.load(f"./runs/{args.exp_name}/metrics.npy")

    mae, me, rmse, percent_rmse, rmspe, r2 = data

    print("Metrics:\n")
    print(f"MAE: {mae}")
    print(f"Mean Error: {me}")
    print(f"RMSE: {rmse}")
    print(f"% RMSE: {percent_rmse}")
    print(f"RMSPE: {rmspe}")