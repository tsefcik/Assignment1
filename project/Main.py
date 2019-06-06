from project import Iris as iris
from project import Glass as glass
from project import BreastCancer as bc
from project import Soybean as soy
from project import HouseVotes as votes


def main():
    iris_obj = iris.Iris()
    iris_data = iris_obj.setup_data_iris()
    glass_obj = glass.Glass()
    glass_data = glass_obj.setup_data_glass()
    bc_obj = bc.BreastCancer()
    breast_cancer_data = bc_obj.setup_data_breast_cancer()
    soy_obj = soy.Soybean()
    soy_data = soy_obj.setup_data_soybean()
    votes_obj = votes.HouseVotes()
    votes_data = votes_obj.setup_data_votes()


if __name__ == "__main__":
    main()
