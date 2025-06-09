import pandas as pd

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(len(df), "rows in the dataset")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    print("size of dataframe:", len(df), "rows")

    return df

def run(year, month):
    read_dataframe(year=year, month=month)

    # if month == 12:
    #     year += 1
    #     month = 1
    # else:
    #     month += 1

    # df_val = read_dataframe(year=year, month=month + 1)

    # X_train, dv = create_X(df_train)
    # X_val, _ = create_X(df_val, dv=dv)

    # y_train = df_train['duration'].values
    # y_val = df_val['duration'].values

    # run_id = train_model(X_train, y_train, X_val, y_val, dv)

    # return run_id

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model to predict taxi trip duration")
    parser.add_argument('--year', type=int, default=2023, help='Year of the data')
    parser.add_argument('--month', type=int, default=3, help='Month of the data')
    args = parser.parse_args()

    run(year=args.year, month=args.month)
    #save run_id to a file
    # with open("run_id.txt", "w") as f:
    #     f.write(run_id)