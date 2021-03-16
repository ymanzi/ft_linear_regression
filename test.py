from mylinearregression import *

# data_array = read_csv_get_data_split("data.csv")
# save_data_split("data_split_save.npy", data_array)

data_array = get_data_split("data_split_save.npy")

data_array[0] = minmax_normalization(data_array[0])
data_array[2] = minmax_normalization(data_array[2])

x_train = data_array[0].reshape(-1, 1)
y_train = data_array[1].reshape(-1, 1)

x_test = data_array[2].reshape(-1, 1)
y_test = data_array[3].reshape(-1, 1)

# x_train = add_polynomial_features(x_train, 5)



theta = np.zeros(x_train.shape[1] + 1)
mlr = MyLinearRegression(theta)
# print(mlr.cost_(x_train, y_train))
# mlr.fit_(x_train, y_train, alpha=0.0001, n_cycle=5000000) # power = 1
mlr.fit_(x_train, y_train, alpha=1e-3, n_cycle=1000000)
mlr.plot_(x_train, y_train)
mlr.plot_(x_test, y_test)
print("train: {}".format(mlr.cost_(x_train, y_train)))
print("test: {}".format(mlr.cost_(x_test, y_test)))