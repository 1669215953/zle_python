import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

token = 'c82e2dcc0a6f8e42e031deac7fcbdbfc28d1d4b7a6852cc1a71dc389'
ts.set_token(token)
pro = ts.pro_api()  # 接入api

class SVR_Predict:
    stock_code = ''
    tsData = pd.DataFrame()

    def __init__(self, stock_code):
        self.stock_code = stock_code

    def date_setting(self, start_date, end_date):
        self.tsData = pro.daily(ts_code=self.stock_code, start_date=start_date, end_date=end_date)
        self.tsData = self.tsData.sort_index(ascending=True).reset_index()

    def makePrediction(self, node):
        # 创建数据框
        new_data = pd.DataFrame(index=range(0, len(self.tsData)), columns=['Date', 'Close'])
        for i in range(0, len(self.tsData)):
            new_data['Date'][i] = self.tsData.index[i]
            new_data['Close'][i] = self.tsData["close"][i]
        # 设置索引
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)

        # 创建训练集和验证集
        dataset = new_data.values
        train = dataset[0:node, :]
        valid = dataset[node:, :]

        # 将数据集转换为x_train和y_train
        scaler = MinMaxScaler(feature_range=(0, 0.8))
        scaled_data = scaler.fit_transform(dataset)
        x_train, y_train = [], []
        for i in range(60, len(train)):
            x_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))

        # 创建和拟合SVR模型
        model = SVR(kernel='linear', C=10, gamma=0.1)
        model.fit(x_train, y_train)

        # 使用SVR模型进行预测
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price.reshape(-1, 1))

        # 作图
        train = new_data[:node]
        valid = new_data[node:]
        print('valid长度是：' + str(len(valid)))
        print(len(closing_price))
        print(valid)
        valid['Predictions'] = closing_price
        print(valid)
        plt.plot(train['Close'], label='训练集')
        plt.plot(valid['Close'], label='真实值')
        plt.plot(valid['Predictions'], label='预测值')
        plt.show()

    def print(self):
        print(self.tsData)

a = SVR_Predict('000001.SZ')
a.date_setting(start_date='20220101', end_date='20221231')
a.makePrediction(130)