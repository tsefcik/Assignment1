import pandas as pd
from sklearn import preprocessing


class Iris:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    class_we_want = "Iris-virginica"

    def setup_data_iris(self):
        # Read in data file and turn into data structure
        iris = pd.read_csv("data/iris.data",
                           sep=",",
                           header=0,
                           names=self.iris_names)
        print("Initial data frame:\n")
        print(iris)  # Show data

        # Make categorical column a binary for the class we want to use
        for index, row in iris.iterrows():
            if iris["class"][index] == self.class_we_want:
                iris.at[index, "class"] = 1
            else:
                iris.at[index, "class"] = 0
        print("Updated data frame with categorical change:\n")
        print(iris)  # Show data

        # Get copy of data with columns that will be normalized
        new_iris = iris[iris.columns[0:4]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        iris_scaled_data = scaler.fit_transform(new_iris)
        # Remove "class" column for now since that column will not be normalized
        self.iris_names.remove("class")
        iris_scaled_data = pd.DataFrame(iris_scaled_data, columns=self.iris_names)
        # Add "class" column back to our column list
        self.iris_names.append("class")

        # Add "class" column into normalized data structure, then categorize it into integers
        iris_scaled_data["class"] = iris[["class"]]
        print("Scaled data:\n")
        print(iris_scaled_data)  # Show data

        # Get mean of each column that will help determine what binary value to turn each into
        iris_means = iris_scaled_data.mean()
        print("Means:\n")
        print(iris_means)  # Show means

        # Make categorical column a binary for the class we want to use
        for index, row in iris_scaled_data.iterrows():
            for column in self.iris_names:
                # If the data value is greater than the mean of the column, make it a 1
                if iris_scaled_data[column][index] > iris_means[column]:
                    iris_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    iris_scaled_data.at[index, column] = 0

        print("One hot encoded data frame:\n")
        print(iris_scaled_data)  # Show data

        # Return one hot encoded data frame
        return iris_scaled_data
