from lib.mylinearregression import *
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("incorrect number of arguments")
    else:
        f = pd.read_csv("resources/data.csv")
        x_train = minmax_normalization(np.array(f['km'])).reshape(-1, 1)
        y_train = np.array(f['price']).reshape(-1, 1)

        theta = np.zeros(x_train.shape[1] + 1)
        mlr = MyLinearRegression(theta)
        print("Cost Before train: {}".format(mlr.cost_(x_train, y_train)))
        mlr.fit_(x_train, y_train, alpha=1e-3, n_cycle=1000000)

        df = pd.DataFrame({"Thetas":mlr.theta})
        df.to_csv("resources/theta.csv", index=False)
        print("Cost After train: {}".format(mlr.cost_(x_train, y_train)))

        # mlr.plot_(x_train, y_train)