# import pandas
# import numpy
# import os


# def regression_s(dirName, fileName):
# 	data = pandas.read_csv(dirName + '/' + fileName)
# 	data_x = numpy.array(data['name'])
# 	data_y = numpy.array(data['data'])
# 	print("File", dirName + '/' + fileName + ':')
# 	for name, data in zip(data_x, data_y):
# 		print(name, data)

# for root, _, files in os.walk('./data'):
# 	print(root, files)

# 	for fileIns in files:
# 		print("Current file: " + root + '/' + fileIns)

# 		regression_s(root, fileIns)



# from sklearn.model_selection import train_test_split
# x, y = range(10), range(15)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# print(x)
# print(x_train, x_test)

# print(y)
# print(y_train, y_test)

import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(x, np.array([1, 2])) + 3
print(y)

reg = LinearRegression().fit(x, y)
print(reg.score(x, y))

x = np.array([[1, 2], [1, 2], [2, 2], [2, 3]])
reg = LinearRegression().fit(x, y)
print(reg.score(x, y))
