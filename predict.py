import pandas as pd
import numpy as np
import sys

def minmax_normalization(x):
    max = float(np.max(x))
    min = float(np.min(x))
    range = max - min
    return np.divide(np.subtract(x, min), range)

def minmax_denormalization(x, value):
    max = float(np.max(x))
    min = float(np.min(x))
    range = max - min
    return (value - min) / range

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("incorrect number of arguments")
    else:
        theta = np.array(pd.read_csv("resources/theta.csv"))
        mileage = input("Which mileage's price would you like: ")
        f = pd.read_csv("resources/data.csv")
        x_train = np.array(f['km'])
        try:
            mileage = minmax_denormalization(x_train, float(mileage))
            tmp_t0 = theta[0][0]
            tmp_t1 = theta[1][0]

            price = tmp_t0 + mileage * tmp_t1
            print("Predicted Price: {}".format(price))
        except:
            print("Error: You didn't enter a number")