import pandas as pd


def get_data(nrows=10000):
    try:
        df = pd.read_csv('../raw_data/train.csv', nrows = nrows)

    except BaseException:
        url = "s3://wagon-public-datasets/taxi-fare-train.csv"
        df = pd.read_csv(url, nrows=nrows)

    return df

def clean_data(df):
    '''returns a DataFrame without outliers and missing values'''
    df = df.dropna(how='any')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count > 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == "__main__":
  print(get_data(4))
