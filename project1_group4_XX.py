import pandas as pd
import numpy as np
import sklearn

#=================Part 1=================
def part1():
    data1 = sklearn.datasets.load_breast_cancer() # Data set for Part 1

#=================Part 2=================
def part2():
    data2 = sklearn.datasets.fetch_california_housing() # Data set for Part 2

#=================Part 3=================
def part3():
    df = pd.read_csv("Mall_Customers.csv") # Data set for Part 3

#=================Part 4=================
def part4():
    df2 = pd.read_csv("Air_Quality.csv") # Data set for Part 4

#=================Main=================
def main():
    part1()
    part2()
    part3()
    part4()

if __name__ == "__main__":
    main()