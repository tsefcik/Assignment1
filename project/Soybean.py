from builtins import print

import pandas as pd
from sklearn import preprocessing


class Soybean:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    class_we_want = "D3"

    def setup_data_soybean(self):
        soybean_names = []
        # Fill soybean_names with column indexes
        for number in range(0, 36):
            soybean_names.append(str(number))

        # Read in data file and turn into data structure
        soybean = pd.read_csv("data/soybean-small.data",
                              sep=",",
                              header=0,
                              names=soybean_names)
        print("Initial data frame:\n")
        print(soybean)  # Show data

        # Make categorical column a binary for the class we want to use
        for index, row in soybean.iterrows():
            if soybean["35"][index] == self.class_we_want:
                soybean.at[index, "35"] = 1
            else:
                soybean.at[index, "35"] = 0
        print("Updated data frame with categorical change:\n")
        print(soybean)  # Show data

        # Get copy of data with columns that will be normalized
        new_soybean = soybean[soybean.columns[0:35]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        soybean_scaled_data = scaler.fit_transform(new_soybean)
        # Remove "class" column for now since that column will not be normalized
        soybean_names.remove("35")
        soybean_scaled_data = pd.DataFrame(soybean_scaled_data, columns=soybean_names)
        # Add "class" column back to our column list
        soybean_names.append("35")

        # Add "class" column into normalized data structure, then categorize it into integers
        soybean_scaled_data["35"] = soybean[["35"]]
        print("Scaled data:\n")
        print(soybean_scaled_data)  # Show data

        # Get mean of each column that will help determine what binary value to turn each into
        soybean_means = soybean_scaled_data.mean()
        print("Means:\n")
        print(soybean_means)  # Show means

        # Make categorical column a binary for the class we want to use
        for index, row in soybean_scaled_data.iterrows():
            for column in soybean_names:
                # If the data value is greater than the mean of the column, make it a 1
                if soybean_scaled_data[column][index] > soybean_means[column]:
                    soybean_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    soybean_scaled_data.at[index, column] = 0
        print("One hot encoded data frame:\n")
        print(soybean_scaled_data)  # Show data

        # Return one hot encoded data frame
        return soybean_scaled_data
