# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,HuberRegressor,Ridge,Lasso,PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def data_y_process(data):

    new_data = data.drop_duplicates()
    return np.array(new_data["Y"])

def data_x_process(data):
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    setTE = set(data["TERMINALNO"])

    # 设立新数据，用户id，用户行驶时间，用户行驶时间是否拨打电话，高速行驶过程中是否急转弯，
    new_a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    for i in setTE:
        tempdata = data.loc[data["TERMINALNO"] == i]
        tempdata = tempdata.sort_values(["TIME"])

        # 初始化 时间，方向变化
        tempTime = tempdata["TIME"].iloc[0]
        tempdir = tempdata["DIRECTION"].iloc[0]
        tempheight = tempdata["HEIGHT"].iloc[0]

        # 根据时间信息判断最长时间
        maxTime = 0
        maxTimelist = []

        # 用户行驶过程中，打电话危机上升
        phonerisk = 0

        # Direction 突变超过
        dir_risk = 0

        # Height 高度的危险值
        height_risk = 0

        for index, row in tempdata.iterrows():

            # 判断在行驶时候有接通电话
            if row["SPEED"] > 20 and (row["CALLSTATE"] != 4):
                if row["CALLSTATE"] == 0:
                    phonerisk += 0.1
                else:
                    phonerisk += 1

            # 根据时间行驶判断
            if row["TIME"] - tempTime == 60:
                maxTime += 60
                tempTime = row["TIME"]

                # 判断方向是否突变
                if (90 <= abs((row["DIRECTION"] - tempdir))) < 180 and row["SPEED"] > 20:
                    dir_risk += 1
                tempdir = row["DIRECTION"]

                if 10 < row["HEIGHT"] - tempheight and row["SPEED"] > 20:
                    height_risk += 1
                tempheight = row["HEIGHT"]

            elif row["TIME"] - tempTime > 60:
                maxTimelist.append(maxTime)
                maxTime = 0
                tempTime = row["TIME"]

                tempdir = row["DIRECTION"]
                tempheight = row["HEIGHT"]

        speed_max = tempdata["SPEED"].max()
        speed_mean = tempdata["SPEED"].mean()

        latitude_min = tempdata["LATITUDE"].min()
        latitude_max = tempdata["LATITUDE"].max()

        longitude_min = tempdata["LONGITUDE"].min()
        longitude_max = tempdata["LONGITUDE"].max()

        height_mean = tempdata["HEIGHT"].mean()

        maxTimelist.append(maxTime)
        maxTime = max(maxTimelist)
        # print(i,maxTime,phonerisk,dir_risk,height_risk)

        new_a = np.row_stack((new_a, [i,maxTime, phonerisk, dir_risk, height_risk, speed_max, speed_mean,latitude_min,latitude_max,longitude_min,longitude_max,height_mean]))

    return new_a[1:]



def process(xlist,ylist):
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    pp = len(xlist)
    print(len(xlist),len(ylist))
    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for a in range(pp):
            writer.writerow([int(xlist[a]), ylist[a]])


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口

    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)

    train_data_x = data_x_process(train_data.iloc[:,:-1])
    train_data_y = data_y_process(train_data[["TERMINALNO","Y"]])

    test_data_x = data_x_process(test_data)
    # print(test_data_x)

    # ----------------------- 回归树模型 -----------------------
    # tree = DecisionTreeRegressor()
    # tree.fit(train_data_x[:,1:],train_data_y)
    # predict_y = tree.predict(test_data_x[:,1:])
    # print(predict_y)
    # *********************************************************

    # -------------------- 随机森林回归模型 ---------------------
    regr = RandomForestRegressor(max_features="log2",max_depth=4,random_state=22,n_jobs=-1)
    regr.fit(train_data_x[:,1:],train_data_y)
    predict_y = regr.predict(test_data_x[:, 1:])
    # print(predict_y)
    print(np.median(predict_y))
    print(predict_y.mean())
    # *********************************************************

    predic_x = test_data_x[:,0]

    # print(predic_x)
    process(predic_x,predict_y)
