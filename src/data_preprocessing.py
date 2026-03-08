import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df.drop("customerID", axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
    df = pd.get_dummies(df, drop_first=True)
    return df