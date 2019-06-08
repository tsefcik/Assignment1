import pandas as pd
from sklearn import preprocessing


class BreastCancer:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                           "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                           "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    class_we_want = 4

    def setup_data_breast_cancer(self):
        # Read in data file and turn into data structure
        breast_cancer = pd.read_csv("data/breast-cancer-wisconsin.data",
                                    sep=",",
                                    header=0,
                                    names=self.breast_cancer_names)
        print("Initial data frame:\n")
        print(breast_cancer)  # Show data
        print()

        # Keep track of rows to drop with missing data
        rows_to_drop = []

        # Find rows with values not filled in
        for column in self.breast_cancer_names:
            for index, row in breast_cancer.iterrows():
                if row[column] == "?":
                    rows_to_drop.append(index)

        # Drop rows without full data
        breast_cancer = breast_cancer.drop(rows_to_drop, axis=0)
        # Show data
        print("Updated data frame without missing data:\n")
        print(breast_cancer)
        print()

        # Make categorical column a binary for the class we want to use
        for index, row in breast_cancer.iterrows():
            if breast_cancer["Class"][index] == self.class_we_want:
                breast_cancer.at[index, "Class"] = 1
            else:
                breast_cancer.at[index, "Class"] = 0
        print("Updated data frame with categorical change:\n")
        print(breast_cancer)  # Show data
        print()

        # Get copy of data with columns that will be normalized
        new_breast_cancer = breast_cancer[breast_cancer.columns[1:10]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        breast_cancer_scaled_data = scaler.fit_transform(new_breast_cancer)
        # Remove "Class" column for now since that column will not be normalized
        self.breast_cancer_names.remove("Class")
        # Remove "Sample code number" column for now since that column will not be normalized
        self.breast_cancer_names.remove("Sample code number")
        breast_cancer_scaled_data = pd.DataFrame(breast_cancer_scaled_data, columns=self.breast_cancer_names)
        # Add "class" column back to our column list
        self.breast_cancer_names.append("Class")

        # Add "Class" column into normalized data structure, then categorize it into integers
        breast_cancer_scaled_data["Class"] = breast_cancer[["Class"]]
        print("Scaled data:\n")
        print(breast_cancer_scaled_data)  # Show data
        print()

        # Get mean of each column that will help determine what binary value to turn each into
        breast_cancer_means = breast_cancer_scaled_data.mean()
        print("Means:\n")
        print(breast_cancer_means)  # Show means
        print()

        # Make categorical column a binary for the class we want to use
        for index, row in breast_cancer_scaled_data.iterrows():
            for column in self.breast_cancer_names:
                # If the data value is greater than the mean of the column, make it a 1
                if breast_cancer_scaled_data[column][index] > breast_cancer_means[column]:
                    breast_cancer_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    breast_cancer_scaled_data.at[index, column] = 0
        print("One hot encoded data frame:\n")
        print(breast_cancer_scaled_data)  # Show data
        print()

        # Add column back in bc it was throwing an error on the iterations of running the winnow alg
        self.breast_cancer_names.insert(0, "Sample code number")

        # Return one hot encoded data frame
        return breast_cancer_scaled_data
