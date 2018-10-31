#!/usr/bin/env python3

import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse
conf = 0.95

parser = argparse.ArgumentParser(description="""
Little utility created by paulostro to draw result graphs with confidence interval.

Input files are .csv with each line being the result of a run. This scripts \
computes the mean of the runs and the confidence interval and draws it using \
matplotlib.pyplot .

The first line can correspond to the graduation on the X axis, in this case use the \
argument -x.

exemple:
python draw_results.py -x --Xlabel "training sample" --Ylabel "error rate" \
--title "relevant title" -n "courbe1" "courbe2" \
 -f data/data1.csv data/data2.csv
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-x', '--X', action="store_true")
parser.add_argument('--title', type=str, default='')
parser.add_argument('--Xlabel', type=str, default='')
parser.add_argument('--Ylabel', type=str, default='')
parser.add_argument('-n', '--plotnames', nargs='+')
parser.add_argument('-f', '--filenames', nargs='+')
args = parser.parse_args()
filenames = args.filenames
X = args.X

for filename in filenames:
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        results = list(csv_reader)
        print([i for i, x in enumerate(results) if len(x) != len(results[0])])
        results = np.array(results).astype(np.float)
    if X:
        x = results[0]
        results = results[1:]
    else:
        x = np.arange(1, results.shape[1] + 1)

    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    confidence = std * 2.0 / (len(std)**(0.5))
    print(len(results))

    plt.errorbar(x, mean, yerr=confidence)

plt.legend(args.plotnames)
plt.title(args.title)
plt.xlabel(args.Xlabel)
plt.ylabel(args.Ylabel)
plt.show()
