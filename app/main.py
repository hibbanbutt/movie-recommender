import pandas as pd


def load_data(filename):
    return pd.read_csv(filename, sep='::', encoding='latin-1', header=None, engine='python')


def main():
    print(load_data('data/movies.dat'))


if __name__ == "__main__":
    main()
