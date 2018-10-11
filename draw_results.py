import numpy as np 
import csv
import sys
import scipy.stats as ss 
import matplotlib.pyplot as plt

conf = 0.95
filename = sys.argv[1]

with open(filename, 'r') as f:
    csv_reader = csv.reader(f)
    results = list(csv_reader)
    print([i for i,x in enumerate(results) if len(x) != len(results[0])])
    results = np.array(results).astype(np.float)

x = np.arange(1, results.shape[1] + 1) 

mean = np.mean(results, axis=0)
std = np.std(results, axis=0)

confidence = std * 3.0 / (len(std)**(0.5))
confidence[(x % 100) != 0] = 0

plt.errorbar(x, mean, yerr=confidence)
plt.show()