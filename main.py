import pandas as pd
import os

train = "cities_dataset_10/train_validation"
test = "cities_dataset_10/test"

def make_df(filename):
    dic = {}
    for file in os.listdir(filename):
        d = os.path.join(filename, file)
        if os.path.isdir(d):
            dic[file] = []
            for pic in os.listdir(d):
                name = os.path.join(d, pic)
                dic[file].append(name)

    smallest = len(dic["Amsterdam"])      
    for bit in dic.keys():
        if len(dic[bit]) < smallest:
            smallest = len(dic[bit])
    for bit in dic.keys():
        while len(dic[bit]) > smallest:
            dic[bit].pop(0)

    return pd.DataFrame(data=dic)


train_df = make_df(train)
test_df = make_df(test)

print(train_df)