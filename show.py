import pandas as pd
import numpy as np
import sys
from mylinearregression import MyLinearRegression as MLR
from mylinearregression import minmax_normalization

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("incorrect number of arguments")
    else:
        theta = np.array(pd.read_csv("resources/theta.csv"))
        f = pd.read_csv("resources/data.csv")
        x_train = minmax_normalization(np.array(f['km'])).reshape(-1, 1)
        y_train = np.array(f['price']).reshape(-1, 1)
        mlr = MLR(np.zeros(x_train.shape[1] + 1))
        mlr.plot_(x_train, y_train)
        mlr = MLR(theta)
        mlr.plot_(x_train, y_train)