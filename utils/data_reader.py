
import pandas as pd

from utils.constants import CASE_STUDY_CSV_DIR


class DataReader:
    """Provides access to csv data."""
    columns = []

    def __init__(self):
        pass

    @staticmethod
    def get_data() -> pd.DataFrame:
        """Retrieves CSV file's case study data as Pandas DataFrame."""
        print("Retrieving case study data from: {}".format(CASE_STUDY_CSV_DIR))
        return pd.read_csv(CASE_STUDY_CSV_DIR, delimiter=',')
