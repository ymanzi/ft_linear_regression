import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle

def read_csv_get_data_split(filename):
	f = pd.read_csv(filename)
	x = np.array(f['km'])
	y = np.array(f['price'])
	return data_spliter(x, y, 0.8) # we take 60% of the data for the training

def save_data_split(filename, data_split):
	with open(filename, 'wb') as f:
		np.save(f, data_split, allow_pickle=True)

def get_data_split(filename):
	with open(filename, 'rb') as f:
		ret = np.load(f, allow_pickle=True)
	return ret

def minmax_normalization(x):
	max = float(np.max(x))
	min = float(np.min(x))
	range = max - min
	return np.divide(np.subtract(x, min), range)

def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float):
	"""
		return x_train y_train x_test y_test
	"""
	if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
		return None
	random_zip = list(zip(x.tolist(), y))
	shuffle(random_zip)
	new_x = []
	new_y = []
	for e1, e2 in random_zip:
		new_x.append(e1)
		new_y.append(e2)
	new_x = np.array(new_x)
	new_y = np.array(new_y)
	proportion_position = int(x.shape[0] * proportion)
	ret_array = []
	ret_array.append(new_x[:proportion_position])
	ret_array.append(new_y[:proportion_position])
	ret_array.append(new_x[proportion_position:])
	ret_array.append(new_y[proportion_position:])
	return np.array(ret_array, dtype=np.ndarray)

class MyLinearRegression():
	"""
	Description:
		My personnal linear regression class to fit like a boss.
	"""
	def __init__(self,  theta, alpha=0.001, n_cycle=10000):
		self.alpha = alpha
		self.n_cycle = n_cycle
		if isinstance(theta, list):
			theta = np.array(theta)
		if theta.ndim != 1:
			theta = np.array([elem for lst in theta for elem in lst])
		self.theta = theta

	def add_intercept(self, x: np.ndarray):
		"""
				Adds a column of 1's to the non-empty numpy.ndarray x.
		"""
		if x.size == 0:
			return None
		return np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x))

	def predict_(self, x: np.ndarray):
		"""
			predict the points according to the theta values
		"""
		x_plus = self.add_intercept(x)
		return x_plus.dot(self.theta).reshape(-1, 1)

	def cost_elem_(self, x: np.ndarray, y: np.ndarray):
		"""
			create an array with difference between the real values and the predicted ones
		"""
		predicted_y = self.predict_(x)
		ret = np.array([(e1 - e2)**2 / (2 * y.size) \
			for e1, e2 in zip(predicted_y, y)])
		if ret.shape[1] == 1:
			ret = np.array([elem for lst in ret for elem in lst])
		return ret

	def cost_(self, x: np.ndarray, y: np.ndarray):
		"""
			calculate the cost (the precision) of the prediction in comparaison of the real ones
			sum((elem_predicted - elem_expected)**2) / (2 * size_of_array)
		"""
		return sum(self.cost_elem_(x, y).tolist())

	def fit_(self, x: np.ndarray, y: np.ndarray, alpha = 0.0001, n_cycle = 10000):
		"""
			Will train our algorythm following the gradient to adjust the teta according to our dataset
			the gradient is the derivation of the predicted equation in every direction
		"""
		if y.ndim > 1:
			y = np.array([elem for lst in y for elem in lst])
		x_plus = self.add_intercept(x)
		for i in range(n_cycle):
			x_theta = x_plus.dot(self.theta)  # Predicted values
			x_theta_minus_y = np.subtract(x_theta, y)
			gradient = x_plus.transpose().dot(x_theta_minus_y) / y.shape[0]
			self.theta = np.subtract(self.theta, np.multiply(alpha, gradient))
		return self.theta
	
	def mse_(self, x: np.ndarray, y: np.ndarray):
		predicted_y = self.predict_(x)
		ret = np.array([(e1 - e2)**2 / (y.size) \
			for e1, e2 in zip(predicted_y, y)])
		if ret.shape[1] == 1:
			ret = np.array([elem for lst in ret for elem in lst])
		return sum(ret)

	def plot_(self, x: np.ndarray, y: np.ndarray):
		predicted_values = self.predict_(x)
		plt.scatter(x, y, color="orange") #draw the multiple points
		cost = self.cost_(x, y)
		plt.plot(x, predicted_values, color="blue", marker="o")
		plt.xlabel("Normalized KM")
		plt.ylabel("Price")
		title = "Cost : " + str(cost)[:9]
		plt.title(title)
		plt.show()

	def hist_(self, x: np.ndarray, y: np.ndarray):
		predicted_values = self.predict_(x)
		plt.hist(y) #draw the multiple points
		cost = self.cost_(x, y)
		plt.hist(predicted_values)
		plt.xlabel("Normalized KM")
		plt.ylabel("Price")
		title = "Cost : " + str(cost)[:9]
		plt.title(title)
		plt.show()



	

# X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
# Y = np.array([[23.], [48.], [218.]])
# mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

# mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
# print(mylr.cost_(X,Y))