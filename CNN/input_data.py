import pandas as pd
import numpy as np

def get_training_data(data_dir):
	df = pd.read_csv(data_dir)

	X = df.iloc[:, 1:].values
	X = X.astype(np.float32)
	Y = df.iloc[:, 0].values

	return X, Y
