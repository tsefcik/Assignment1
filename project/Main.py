from builtins import print

from project import Iris as iris
from project import Glass as glass
from project import BreastCancer as bc
from project import Soybean as soy
from project import HouseVotes as votes
from project import Algorithms as alg


def run_winnow_iris():
    success_rate = 0

    for index in range(0, 50):
        iris_obj = iris.Iris()
        iris_data = iris_obj.setup_data_iris()
        winnow = alg.Algorithms()
        iris_data_train = iris_data.sample(frac=.667)
        iris_data_test = iris_data.drop(iris_data_train.index)
        trainer_classifier = winnow.winnow2(iris_data_train.iloc[:, 0:4], iris_data_train["class"], 2, .5, 1)
        test_classifier = winnow.winnow2_test(iris_data_test.iloc[:, 0:4], 2, trainer_classifier)
        success_rate = success_rate + winnow.compare_prediction(test_classifier, iris_data_test["class"])
    success_rate = success_rate / 50
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")


def run_winnow_glass():
    success_rate = 0

    for index in range(0, 50):
        glass_obj = glass.Glass()
        glass_data = glass_obj.setup_data_glass()
        winnow = alg.Algorithms()
        glass_data_train = glass_data.sample(frac=.667)
        glass_data_test = glass_data.drop(glass_data_train.index)
        trainer_classifier = winnow.winnow2(glass_data_train.iloc[:, 0:9], glass_data_train["Type of glass"], 1, 10, 1)
        test_classifier = winnow.winnow2_test(glass_data_test.iloc[:, 0:9], 2, trainer_classifier)
        success_rate = success_rate + winnow.compare_prediction(test_classifier, glass_data_test["Type of glass"])
    success_rate = success_rate / 50
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")


def run_winnow_bc():
    success_rate = 0

    for index in range(0, 50):
        bc_obj = bc.BreastCancer()
        bc_data = bc_obj.setup_data_breast_cancer()
        winnow = alg.Algorithms()
        bc_data_train = bc_data.sample(frac=.667)
        bc_data_test = bc_data.drop(bc_data_train.index)
        trainer_classifier = winnow.winnow2(bc_data_train.iloc[:, 0:10], bc_data_train["Class"], 1, .5, 1)
        test_classifier = winnow.winnow2_test(bc_data_test.iloc[:, 0:10], 2, trainer_classifier)
        success_rate = success_rate + winnow.compare_prediction(test_classifier, bc_data_test["Class"])
    success_rate = success_rate / 50
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")


def run_winnow_soybean():
    success_rate = 0

    for index in range(0, 50):
        soybean_obj = soy.Soybean()
        soybean_data = soybean_obj.setup_data_soybean()
        winnow = alg.Algorithms()
        soybean_data_train = soybean_data.sample(frac=.667)
        soybean_data_test = soybean_data.drop(soybean_data_train.index)
        trainer_classifier = winnow.winnow2(soybean_data_train.iloc[:, 0:36], soybean_data_train["35"], 3, 10, 1)
        test_classifier = winnow.winnow2_test(soybean_data_test.iloc[:, 0:36], 3, trainer_classifier)
        success_rate = success_rate + winnow.compare_prediction(test_classifier, soybean_data_test["35"])
    success_rate = success_rate / 50
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")


def run_winnow_votes():
    success_rate = 0

    for index in range(0, 50):
        votes_obj = votes.HouseVotes()
        votes_data = votes_obj.setup_data_votes()
        winnow = alg.Algorithms()
        votes_data_train = votes_data.sample(frac=.667)
        votes_data_test = votes_data.drop(votes_data_train.index)
        trainer_classifier = winnow.winnow2(votes_data_train.iloc[:, 1:17], votes_data_train["class"], 3, 10, 1)
        test_classifier = winnow.winnow2_test(votes_data_test.iloc[:, 1:17], 3, trainer_classifier)
        success_rate = success_rate + winnow.compare_prediction(test_classifier, votes_data_test["class"])
    success_rate = success_rate / 50
    print("Average Winnow-2 success rate is: " + str(success_rate) + "%")


def run_naive_bayes_iris():
    success_rate = 0

    # for index in range(0, 50):
    iris_obj = iris.Iris()
    iris_data = iris_obj.setup_data_iris()
    naive = alg.Algorithms()
    iris_data_train = iris_data.sample(frac=.667)
    iris_data_test = iris_data.drop(iris_data_train.index)
    trainer_classifier = naive.naive_bayes_train(iris_data_train.iloc[:, 0:4], iris_data_train["class"])
    test_classifier = naive.naive_bayes_test(iris_data_test.iloc[:, 0:4], trainer_classifier)
    success_rate = naive.compare_prediction(test_classifier, iris_data_test["class"])

#   success_rate = success_rate / 50
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")


def main():
    # Winnow-2
    # run_winnow_iris()
    # run_winnow_glass()
    # run_winnow_bc()
    # run_winnow_soybean()
    # run_winnow_votes()

    # Naive Bayes
    run_naive_bayes_iris()


if __name__ == "__main__":
    main()
