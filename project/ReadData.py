import pandas as pd
import numpy as np
from sklearn import preprocessing


class Data:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    glass_names = ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]
    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                           "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                           "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

    def setup_data_glass(self):
        # Read in data file and turn into data structure
        glass = pd.read_csv("data/glass.data",
                            sep=",",
                            header=0,
                            names=self.glass_names)

        # Get copy of data with columns that will be normalized
        new_glass = glass[glass.columns[1:11]]

        # Normalize data with MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        glass_scaled_data = scaler.fit_transform(new_glass)
        # Remove "Id number" column since it won't be normalized
        self.glass_names.remove("Id number")
        glass_scaled_data = pd.DataFrame(glass_scaled_data, columns=self.glass_names)

        # Return scaled data structure for binning
        return glass_scaled_data

    def setup_data_iris(self):
        # Read in data file and turn into data structure
        iris = pd.read_csv("data/iris.data",
                           sep=",",
                           header=0,
                           names=self.iris_names)

        # Get copy of data with columns that will be normalized
        new_iris = iris[iris.columns[0:4]]

        # Normalize data with MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        iris_scaled_data = scaler.fit_transform(new_iris)
        # Remove "class" column for now since that column will not be normalized
        self.iris_names.remove("class")
        iris_scaled_data = pd.DataFrame(iris_scaled_data, columns=self.iris_names)
        # Add "class" column back to our column list
        self.iris_names.append("class")

        # Add "class" column into normalized data structure, then categorize it into integers
        iris_scaled_data["class"] = iris[["class"]]
        iris_scaled_data["class"] = pd.factorize(iris_scaled_data["class"])[0]

        # Return scaled data structure for binning
        return iris_scaled_data

    def setup_data_breast_cancer(self):
        # Read in data file and turn into data structure
        breast_cancer = pd.read_csv("data/breast-cancer-wisconsin.data",
                                    sep=",",
                                    header=0,
                                    names=self.breast_cancer_names)

        # Get copy of data with columns that will be normalized
        new_breast_cancer = breast_cancer[breast_cancer.columns[1:10]]

        # Normalize data with MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        glass_scaled_data = scaler.fit_transform(new_breast_cancer)
        # Remove "Sample code number" column for now since that column will not be normalized
        self.breast_cancer_names.remove("Sample code number")
        breast_cancer_scaled_data = pd.DataFrame(glass_scaled_data, columns=self.breast_cancer_names)

        # Return scaled data structure for binning
        return breast_cancer_scaled_data

    def setup_data_soybean(self):
        soybean = pd.read_csv("data/soybean-small.data",
                              sep=",",
                              header=0,
                              names=None)

        print(soybean)

    # Bin dataframe
    def bin(self, dataframe, column_names):
        # Arbitrary default picked bins for data scaled from 0 to 1
        bins = [0, .25, .50, .75, 1]

        # Loop through column list and bin data
        for index in column_names:
            # If column name is "class", use special bin assignment
            if index == "class":
                # Bin data for iris "class" category
                dataframe[index + " bin"] = pd.cut(dataframe[index], bins=[0, 1, 2], include_lowest=True)
            else:
                # Bin data
                dataframe[index + ' bin'] = pd.cut(dataframe[index], bins, include_lowest=True)

        print(dataframe)

        # Return binned data structure
        return dataframe


data = Data()
# glass = data.setup_data_glass()
# glass_binned = data.bin(glass, data.glass_names)
# iris = data.setup_data_iris()
# iris_binned = data.bin(iris, data.iris_names)
breast_cancer = data.setup_data_breast_cancer()
breast_cancer_binned = data.bin(breast_cancer, data.breast_cancer_names)
# soybean = data.setup_data_soybean()
# soybean_binned = data.bin(soybean, data.soybean_names)

