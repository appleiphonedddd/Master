#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_accuracy_multi.py path/to/rs_test_acc1.csv [path/to/rs_test_acc2.csv ...]")
        sys.exit(1)

    csv_paths = sys.argv[1:]
    plt.figure()

    # Color cycle from matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for idx, csv_path in enumerate(csv_paths):
        if not os.path.isfile(csv_path):
            print(f"[Warning] File not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        label = os.path.splitext(os.path.basename(csv_path))[0]
        color = colors[idx % len(colors)]
        plt.plot(df['round'], df['test_acc'],label=label, color=color)

    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    png_path = 'Result.png'
    plt.savefig(png_path)
    print(f"[Info] Saved comparison plot to {png_path}")

if __name__ == "__main__":
    main()
