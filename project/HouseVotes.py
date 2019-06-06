import pandas as pd
from sklearn import preprocessing


class HouseVotes:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    votes_names = ["class", "handicapped", "water", "adoption", "physician", "el-salvador", "religious",
                   "anti", "aid", "mx", "immigration", "synfuels", "education", "superfund", "crime",
                   "duty-free", "export"]
    class_we_want = "republican"

    def setup_data_votes(self):
        # Read in data file and turn into data structure
        votes = pd.read_csv("data/house-votes-84.data",
                            sep=",",
                            header=0,
                            names=self.votes_names)
        print("Initial data frame:\n")
        print(votes)  # Show data

        # Keep track of rows to drop with missing data
        rows_to_drop = []

        # Find rows with values not filled in
        for column in self.votes_names:
            for index, row in votes.iterrows():
                if row[column] == "?":
                    rows_to_drop.append(index)

        # Drop rows without full data
        votes = votes.drop(rows_to_drop, axis=0)
        # Show data
        print("Updated data frame without missing data:\n")
        print(votes)

        # Make categorical column a binary for the class we want to use
        for index, row in votes.iterrows():
            if votes["class"][index] == self.class_we_want:
                votes.at[index, "class"] = 1
            else:
                votes.at[index, "class"] = 0

        # Make categorical column a binary for the class we want to use
        for column in self.votes_names:
            for index, row in votes.iterrows():
                if votes[column][index] == "y":
                    votes.at[index, column] = 1
                else:
                    votes.at[index, column] = 0
        print("Updated data frame with categorical change:\n")
        print(votes)  # Show data

        # Get copy of data with columns that will be normalized
        new_votes = votes[votes.columns[1:17]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        votes_scaled_data = scaler.fit_transform(new_votes)
        # Remove "class" column for now since that column will not be normalized
        self.votes_names.remove("class")
        votes_scaled_data = pd.DataFrame(votes_scaled_data, columns=self.votes_names)
        # Add "class" column back to our column list
        self.votes_names.append("class")

        # Add "class" column into normalized data structure, then categorize it into integers
        votes_scaled_data["class"] = votes[["class"]]
        print("Scaled data:\n")
        print(votes_scaled_data)  # Show data

        # Get mean of each column that will help determine what binary value to turn each into
        votes_means = votes_scaled_data.mean()
        print("Means:\n")
        print(votes_means)  # Show means

        # Make categorical column a binary for the class we want to use
        for index, row in votes_scaled_data.iterrows():
            for column in self.votes_names:
                # If the data value is greater than the mean of the column, make it a 1
                if votes_scaled_data[column][index] > votes_means[column]:
                    votes_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    votes_scaled_data.at[index, column] = 0
        print("One hot encoded data frame:\n")
        print(votes_scaled_data)  # Show data

        # Return one hot encoded data frame
        return votes_scaled_data
