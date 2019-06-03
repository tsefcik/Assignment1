import pandas as pd

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

glass_names = ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]

breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                       "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                       "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]

glass = pd.read_csv("data/glass.data",
            sep=",",
            header=0,
            names=glass_names)

breast_cancer = pd.read_csv("data/breast-cancer-wisconsin.data",
                            sep=",",
                            header=0,
                            names=breast_cancer_names)

iris = pd.read_csv("data/iris.data",
                   sep=",",
                   header=0,
                   names=iris_names)

print(glass)
print(breast_cancer)
print(iris)
