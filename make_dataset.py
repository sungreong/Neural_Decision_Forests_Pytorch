import pandas as pd

CSV_HEADER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]

train_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
train_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)

test_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)
train_data.to_csv("./data/adult_train.csv")
train_data.to_csv("./data/adult_test.csv")
