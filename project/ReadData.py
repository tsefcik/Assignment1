import pandas as pd
import numpy as np
from sklearn import preprocessing


class Data:
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    def setup_data_glass(self):
        glass_names = ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]

        glass = pd.read_csv("data/glass.data",
                            sep=",",
                            header=0,
                            names=glass_names)

        print(glass)

        new_glass = glass[glass.columns[1:11]]

        print(new_glass)

        scaler = preprocessing.MinMaxScaler()
        glass_scaled_data = scaler.fit_transform(new_glass)
        glass_names.remove("Id number")
        glass_scaled_data = pd.DataFrame(glass_scaled_data, columns=glass_names)

        print(glass_scaled_data)

    def setup_data_iris(self):
        iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]

        iris = pd.read_csv("data/iris.data",
                           sep=",",
                           header=0,
                           names=iris_names)

        print(iris)
        iris_names_normalize = ["sepal length", "sepal width", "petal length", "petal width"]

        # scaler = preprocessing.MinMaxScaler()
        #
        # scaled_data = scaler.fit_transform(iris.as_matrix(columns=iris_names_normalize))
        # scaled_data = pd.DataFrame(scaled_data, columns=iris_names_normalize)

    def setup_data_breast_cancer(self):
        breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                               "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                               "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

        breast_cancer = pd.read_csv("data/breast-cancer-wisconsin.data",
                                    sep=",",
                                    header=0,
                                    names=breast_cancer_names)

        print(breast_cancer)


data = Data()
data.setup_data_glass()

