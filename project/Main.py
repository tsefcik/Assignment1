from builtins import print

from project import Iris as iris
from project import Glass as glass
from project import BreastCancer as bc
from project import Soybean as soy
from project import HouseVotes as votes
from project import Algorithms as alg


def run_winnow_iris():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        iris_obj = iris.Iris()
        iris_data = iris_obj.setup_data_iris()
        # Start algorithm
        winnow = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        iris_data_train = iris_data.sample(frac=.667)
        iris_data_test = iris_data.drop(iris_data_train.index)
        # Train classifier
        trainer_classifier = winnow.winnow2(iris_data_train.iloc[:, 0:4], iris_data_train["class"], 2, .5, 1)
        # Test classifier
        test_classifier = winnow.winnow2_test(iris_data_test.iloc[:, 0:4], 2, trainer_classifier)
        # Get success rate back
        success_rate = success_rate + winnow.compare_prediction(test_classifier, iris_data_test["class"])
    success_rate = success_rate / 25
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")
    return success_rate


def run_winnow_glass():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        glass_obj = glass.Glass()
        glass_data = glass_obj.setup_data_glass()
        # Start algorithm
        winnow = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        glass_data_train = glass_data.sample(frac=.667)
        glass_data_test = glass_data.drop(glass_data_train.index)
        # Train classifier
        trainer_classifier = winnow.winnow2(glass_data_train.iloc[:, 0:9], glass_data_train["Type of glass"], 1, 10, 1)
        # Test classifier
        test_classifier = winnow.winnow2_test(glass_data_test.iloc[:, 0:9], 2, trainer_classifier)
        # Get success rate back
        success_rate = success_rate + winnow.compare_prediction(test_classifier, glass_data_test["Type of glass"])
    success_rate = success_rate / 25
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")
    return success_rate


def run_winnow_bc():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        bc_obj = bc.BreastCancer()
        bc_data = bc_obj.setup_data_breast_cancer()
        # Start algorithm
        winnow = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        bc_data_train = bc_data.sample(frac=.667)
        bc_data_test = bc_data.drop(bc_data_train.index)
        # Train classifier
        trainer_classifier = winnow.winnow2(bc_data_train.iloc[:, 0:10], bc_data_train["Class"], 1, .5, 1)
        # Test classifier
        test_classifier = winnow.winnow2_test(bc_data_test.iloc[:, 0:10], 2, trainer_classifier)
        # Get success rate back
        success_rate = success_rate + winnow.compare_prediction(test_classifier, bc_data_test["Class"])
    success_rate = success_rate / 25
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")
    return success_rate


def run_winnow_soybean():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        soybean_obj = soy.Soybean()
        soybean_data = soybean_obj.setup_data_soybean()
        # Start algorithm
        winnow = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        soybean_data_train = soybean_data.sample(frac=.667)
        soybean_data_test = soybean_data.drop(soybean_data_train.index)
        # Train classifier
        trainer_classifier = winnow.winnow2(soybean_data_train.iloc[:, 0:36], soybean_data_train["35"], 3, 10, 1)
        # Test classifier
        test_classifier = winnow.winnow2_test(soybean_data_test.iloc[:, 0:36], 3, trainer_classifier)
        # Get success rate back
        success_rate = success_rate + winnow.compare_prediction(test_classifier, soybean_data_test["35"])
    success_rate = success_rate / 25
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")
    return success_rate


def run_winnow_votes():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        votes_obj = votes.HouseVotes()
        votes_data = votes_obj.setup_data_votes()
        # Start algorithm
        winnow = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        votes_data_train = votes_data.sample(frac=.667)
        votes_data_test = votes_data.drop(votes_data_train.index)
        # Train classifier
        trainer_classifier = winnow.winnow2(votes_data_train.iloc[:, 1:17], votes_data_train["class"], 3, 10, 1)
        # Test classifier
        test_classifier = winnow.winnow2_test(votes_data_test.iloc[:, 1:17], 3, trainer_classifier)
        # Get success rate back
        success_rate = success_rate + winnow.compare_prediction(test_classifier, votes_data_test["class"])
    success_rate = success_rate / 25
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")
    return success_rate


def run_naive_bayes_iris():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        iris_obj = iris.Iris()
        iris_data = iris_obj.setup_data_iris()
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        iris_data_train = iris_data.sample(frac=.667)
        iris_data_test = iris_data.drop(iris_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(iris_data_train.iloc[:, 0:4], iris_data_train["class"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(iris_data_test.iloc[:, 0:4], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, iris_data_test["class"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


def run_naive_bayes_glass():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        glass_obj = glass.Glass()
        glass_data = glass_obj.setup_data_glass()
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        glass_data_train = glass_data.sample(frac=.667)
        glass_data_test = glass_data.drop(glass_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(glass_data_train.iloc[:, 0:9], glass_data_train["Type of glass"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(glass_data_test.iloc[:, 0:9], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, glass_data_test["Type of glass"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


def run_naive_bayes_bc():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        bc_obj = bc.BreastCancer()
        bc_data = bc_obj.setup_data_breast_cancer()
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        bc_data_train = bc_data.sample(frac=.667)
        bc_data_test = bc_data.drop(bc_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(bc_data_train.iloc[:, 0:10], bc_data_train["Class"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(bc_data_test.iloc[:, 0:10], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, bc_data_test["Class"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


def run_naive_bayes_soybean():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        soybean_obj = soy.Soybean()
        soybean_data = soybean_obj.setup_data_soybean()
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        soybean_data_train = soybean_data.sample(frac=.667)
        soybean_data_test = soybean_data.drop(soybean_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(soybean_data_train.iloc[:, 0:36], soybean_data_train["35"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(soybean_data_test.iloc[:, 0:36], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, soybean_data_test["35"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


def run_naive_bayes_votes():
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        votes_obj = votes.HouseVotes()
        votes_data = votes_obj.setup_data_votes()
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        votes_data_train = votes_data.sample(frac=.667)
        votes_data_test = votes_data.drop(votes_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(votes_data_train.iloc[:, 1:17], votes_data_train["class"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(votes_data_test.iloc[:, 1:17], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, votes_data_test["class"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


def main():
    # Winnow-2
    winnow_iris = run_winnow_iris()
    winnow_glass = run_winnow_glass()
    winnow_bc = run_winnow_bc()
    winnow_soy = run_winnow_soybean()
    winnow_votes = run_winnow_votes()

    average_winnow_combined = (winnow_iris + winnow_glass + winnow_bc + winnow_soy + winnow_votes) / 5

    # Naive Bayes
    naive_iris = run_naive_bayes_iris()
    naive_glass = run_naive_bayes_glass()
    naive_bc = run_naive_bayes_bc()
    naive_soy = run_naive_bayes_soybean()
    naive_votes = run_naive_bayes_votes()

    average_naive_combined = (naive_iris + naive_glass + naive_bc + naive_soy + naive_votes) / 5

    print("Overall statistics")
    print()
    print("Winnow-2 iris: " + str(winnow_iris) + "%")
    print("Winnow-2 glass: " + str(winnow_glass) + "%")
    print("Winnow-2 breast cancer: " + str(winnow_bc) + "%")
    print("Winnow-2 soybean: " + str(winnow_soy) + "%")
    print("Winnow-2 votes: " + str(winnow_votes) + "%")
    print("Naive Bayes iris: " + str(naive_iris) + "%")
    print("Naive Bayes glass: " + str(naive_glass) + "%")
    print("Naive Bayes breast cancer: " + str(naive_bc) + "%")
    print("Naive Bayes soybean: " + str(naive_soy) + "%")
    print("Naive Bayes votes: " + str(naive_votes) + "%")
    print()
    print("Average Winnow-2 Accuracy: " + str(average_winnow_combined) + "%")
    print("Average Naive Bayes Accuracy: " + str(average_naive_combined) + "%")


if __name__ == "__main__":
    main()
