import numpy as np
import pandas as pd

theta = np.zeros(2)

df = pd.DataFrame({"Thetas":theta})
df.to_csv("resources/theta.csv", index=False)