from sklearn.datasets import load_boston
from model import loadModel
from sklearn.preprocessing import StandardScaler

boston = load_boston()
x_train = boston.data
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
linear_svr = loadModel()
linear_svr_y_predict = linear_svr.predict(x_train)
print(linear_svr_y_predict)