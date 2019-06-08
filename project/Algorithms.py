from builtins import print
import pandas as pd
import numpy as np


class Algorithms:

    def winnow2(self, data, predicted, theta, alpha, initial_weight):
        # Initialize classifier to number of columns and set initial values to initial_weight
        classifier = [initial_weight] * (data.shape[1])

        # Iterate through dataframe rows and get the sum of products with the weights
        for index, row in data.iterrows():
            product = row * classifier
            product_sum = sum(product)

            # If the product sum is greater than theta, assign 1, else assign 0
            if product_sum > theta:
                predict_classifier = 1
            else:
                predict_classifier = 0

            # Act on weights if prediction is not correct
            # We do not need to do anything if is was correct
            if predict_classifier != predicted[index]:
                if predicted[index] == 1:
                    # Perform promotion
                    for index2 in range(0, len(classifier) - 1):
                        if classifier[index2] == 1:
                            classifier[index2] = classifier[index2] * alpha
                else:
                    # Perform demotion
                    for index2 in range(0, len(classifier) - 1):
                        if classifier[index2] == 1:
                            classifier[index2] = classifier[index2] / alpha
        print(classifier)
        return classifier

    def winnow2_test(self, data, theta, classifier):
        # Initialize prediction classifier
        predict_classifier = []

        # Iterate through dataframe rows and get the sum of products with the classifier
        for index, row in data.iterrows():
            product = row * classifier
            product_sum = sum(product)

            # If the product sum is greater than theta, assign 1, else assign 0
            if product_sum > theta:
                guess_classifier = 1
                predict_classifier.append(guess_classifier)
            else:
                guess_classifier = 0
                predict_classifier.append(guess_classifier)

        print(predict_classifier)
        return predict_classifier

    def naive_bayes_train(self, data, predicted):

        col_list = list(data.columns)
        col_list.insert(0, "Overall Mean")
        output = pd.DataFrame(columns=[col_list])
        overall_mean = np.mean(predicted)
        print(overall_mean)

        output.loc[0] = (1-overall_mean)
        output.loc[1] = (1 - overall_mean)
        output.loc[2] = overall_mean
        output.loc[3] = overall_mean

        print(output)

        class_false = data[predicted == 0]
        class_true = data[predicted == 1]
        print(class_false)
        print(class_true)

        new_row = [.5] * (class_false.shape[1])

        class_false = class_false.append(pd.Series(new_row, index=class_false.columns), ignore_index=True)
        class_true = class_true.append(pd.Series(new_row, index=class_true.columns), ignore_index=True)
        print(class_false)
        print(class_true)

        class_false_means = class_false.mean()
        class_true_means = class_true.mean()
        print(class_false_means)
        print(class_true_means)

        adjusted_false = 1 - class_false_means
        print(adjusted_false)

        adjusted_true = 1 - class_true_means
        print(adjusted_true)

        for col in col_list:
            if col != "Overall Mean":
                output.loc[0, col] = adjusted_false[col]
                output.loc[1, col] = class_false_means[col]
                output.loc[2, col] = adjusted_true[col]
                output.loc[3, col] = class_true_means[col]

        print(output)

        return output

    def naive_bayes_test(self, data, mean_matrix):
        predictions = []
        for index, row in data.iterrows():
            class_false = [0] * (len(data.columns) + 1)
            class_false[0] = mean_matrix.iloc[0, 0]
            class_true = [0] * (len(data.columns) + 1)
            class_true[0] = mean_matrix.iloc[0, 0]

            for index2 in range(len(data.columns)):
                if row[index2] == 1:
                    class_false[index2 + 1] = mean_matrix.iloc[1, index2 + 1]
                    class_true[index2 + 1] = mean_matrix.iloc[3, index2 + 1]
                else:
                    class_false[index2 + 1] = mean_matrix.iloc[1, index2 + 1]
                    class_true[index2 + 1] = mean_matrix.iloc[3, index2 + 1]

            class_false_product = np.product(class_false)
            class_true_product = np.product(class_true)

            print(class_false_product)
            print(class_true_product)

            if class_false_product < class_true_product:
                predictions.append(1)
            else:
                predictions.append(0)

        print(predictions)
        print(len(predictions))





    def compare_prediction(self, predict_classier, data):
        success = 0
        for index in range(0, len(predict_classier) - 1):

            if predict_classier[index] == data.values[index]:
                success = success + 1

        success_rate = (success/data.shape[0]) * 100
        print("Success rate is: " + str(success_rate) + "%")

        return success_rate
