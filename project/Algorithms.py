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

    def naive_bayes(self, data, predicted):
        output = pd.DataFrame(index=np.arange(4), columns=np.arange(0))
        overall_mean = np.mean(predicted)
        print(overall_mean)

        output.insert(0, 0, 1-overall_mean, allow_duplicates=True)
        output.insert(1, 1, 1 - overall_mean, allow_duplicates=True)
        output.insert(2, 2, overall_mean, allow_duplicates=True)
        output.insert(3, 3, overall_mean, allow_duplicates=True)

        print(output)

        class_false = data[predicted == 0]
        class_true = data[predicted == 1]
        print(class_false)
        print(class_true)

        # smooth = [.5] * (class_false.shape[1])
        # print(smooth)
        #
        # class_false.append(smooth, ignore_index=True)
        # class_true.append(smooth, ignore_index=True)
        #
        # print(class_false)
        # print(class_true)

    def compare_prediction(self, predict_classier, data):
        success = 0
        for index in range(0, len(predict_classier) - 1):

            if predict_classier[index] == data.values[index]:
                success = success + 1

        success_rate = (success/data.shape[0]) * 100
        print("Success rate is: " + str(success_rate) + "%")

        return success_rate
