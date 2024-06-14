import pandas as pd
from alp import calc_shares, enter_positions, close_positions

from allocations import get_weights


def main():
    dates = pd.read_csv('dates.csv')
    date = dates.iloc[0].values

    dates = dates.iloc[1:]
    dates.to_csv('dates.csv', index=False)

    close_positions()

    weight_dist = get_weights(date[0], date[2], date[1])
    share_dist = calc_shares(weight_dist)

    enter_positions(share_dist)

if __name__ == '__main__':
    main()