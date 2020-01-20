
import pandas as pd
from random import seed
from random import randint

from utils.constants import START, END, SEEDS, NO_TRAIN, NO_TEST


class DataGenerator:
    """Provides seed specific PRNG functionality."""
    columns = []
    data = []

    def __init__(self):
        for i in range(NO_TRAIN):
            self.columns.append('train_' + str(i))
        for i in range(NO_TEST):
            self.columns.append('test_' + str(i))
        self.columns.append('seed')

    def get_data(self) -> pd.DataFrame:
        print("Re-generating case study data.")
        for s in SEEDS:
            seed(s)
            row = []
            for i in range(len(self.columns)-1):
                row.append(randint(START, END))
            row.append(s)
            self.data.append(row)
        return pd.DataFrame(data=self.data, columns=self.columns)
