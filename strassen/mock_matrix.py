import numpy as np
import pandas as pd
import time 

# Generate 1000x1000 random integer matrices
A = np.random.randint(0, 100, size=(1000, 1000))
B = np.random.randint(0, 100, size=(1000, 1000))

# Save to CSV
pd.DataFrame(A).to_csv("A.csv", header=False, index=False)
pd.DataFrame(B).to_csv("B.csv", header=False, index=False)

# test multiply
start_time = time.time()
C = np.dot(A, B)
end_time = time.time()
print(f"Matrix multiplication took {end_time - start_time} seconds.")