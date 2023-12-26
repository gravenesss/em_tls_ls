import numpy as np
import pandas as pd

np.random.seed(1)
random_indices = np.random.permutation(124)
print(random_indices)


# Load the data from the file
file_path = 'data/build_features1.csv'
df = pd.read_csv(file_path)
df_reordered = df.iloc[random_indices]
df_reordered.to_csv(file_path, index=False)
