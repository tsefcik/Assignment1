import pandas as pd
from sklearn import preprocessing


class Glass:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    glass_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]
    class_we_want = 3

    def setup_data_glass(self):
        # Read in data file and turn into data structure
        glass = pd.read_csv("data/glass.data",
                           sep=",",
                           header=0,
                           names=self.glass_names)
        print("Initial data frame:\n")
        print(glass)  # Show data
        print()

        # Make categorical column a binary for the class we want to use
        for index, row in glass.iterrows():
            if glass["Type of glass"][index] > self.class_we_want:
                glass.at[index, "Type of glass"] = 1
            else:
                glass.at[index, "Type of glass"] = 0
        print("Updated data frame with categorical change:\n")
        print(glass)  # Show data
        print()

        # Get copy of data with columns that will be normalized
        new_glass = glass[glass.columns[1:10]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        glass_scaled_data = scaler.fit_transform(new_glass)
        # Remove "Type of glass" column for now since that column will not be normalized
        self.glass_names.remove("Type of glass")
        # Remove "Id number" column for now since that column will not be normalized
        self.glass_names.remove("Id")
        glass_scaled_data = pd.DataFrame(glass_scaled_data, columns=self.glass_names)
        # Add "Type of glass" column back to our column list
        self.glass_names.append("Type of glass")

        # Add "Type of glass" column into normalized data structure, then categorize it into integers
        glass_scaled_data["Type of glass"] = glass[["Type of glass"]]
        print("Scaled data:\n")
        print(glass_scaled_data)  # Show data
        print()

        # Get mean of each column that will help determine what binary value to turn each into
        glass_means = glass_scaled_data.mean()
        print("Means:\n")
        print(glass_means)  # Show means
        print()

        # Make categorical column a binary for the class we want to use
        for index, row in glass_scaled_data.iterrows():
            for column in self.glass_names:
                # If the data value is greater than the mean of the column, make it a 1
                if glass_scaled_data[column][index] > glass_means[column]:
                    glass_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    glass_scaled_data.at[index, column] = 0
        print("One hot encoded data frame:\n")
        print(glass_scaled_data)  # Show data
        print()

        # Add column back in bc it was throwing an error on the iterations of running the winnow alg
        self.glass_names.insert(0, "Id")

        # Return one hot encoded data frame
        return glass_scaled_data
