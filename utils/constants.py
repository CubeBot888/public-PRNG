import os

# Max & Min values of randomly generated numbers
START = 0
END = 1000000
# Random number generator seeds
SEEDS = [1, 34, 65, 87, 2, 8, 12, 0, 90, 56, 78, 11, 43, 6, 7, 5, 10, 99, 100, 101, 3000, 973]
# The number of training, test and total data points.
NO_TRAIN = 40
NO_TEST = 10
NO_POINTS = NO_TEST + NO_TRAIN

PKG_DIR = os.path.abspath('..')
CASE_STUDY_CSV_DIR = os.path.join(PKG_DIR, "data/case_study_data.csv")
