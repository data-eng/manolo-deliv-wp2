import pycatch22
import numpy as np

# TODO READ CSV FILE
tsData = np.array(np.random.randn(7600)) 


# calculate 24 feature values of tsData series:
results = pycatch22.catch22_all(tsData, short_names=True, catch24 = True)

# print 24 features: short_names and values of tsData tsData series:
for (short_name, val) in zip(results['short_names'], results['values']):
    print(f"{short_name}: {val}")