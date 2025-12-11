import numpy as np
import pandas as pd
#load matrix from csv

a = pd.read_csv("matrix_A.csv", header=None)
b = pd.read_csv("matrix_B.csv", header=None)
c = pd.read_csv("matrix_C.csv", header=None)
print(a)
print(b)
c_ = np.dot(a, b)
print(c_)

assert np.array_equal(c.values, c_.astype(int))
print("Matrix multiplication is correct.")