
from utils.latent_state_model import LatentStateModel
from utils.visualisations import display_plots, line_plot_all, line_plot_with_predictions, cross_correlation

# Select use of CSV data or re-generated PRNG data.
USING_CSV = True
if USING_CSV:
    from utils.data_reader import DataReader as Data
else:
    from utils.data_generator import DataGenerator as Data


def main():
    """
    For a given set of pseudo-random number generator (PRNG) seeds, model the function which maps the domain (X) to
    the range(Y), where X is the index of the sequence for the number generated and Y is the output of the PRNG at
    that index.
    """
    # Read or Generate data
    df = Data().get_data()
    # print("First {} row of data:\n{}".format(3, df.head(n=3)))

    # Quickly visualise given (row=integer) or all (raw=None) data.
    line_plot_all(df=df, row=None)
    display_plots()

    # Train and predict each PRNG seed's sequence using an Latent State Model.
    RMSEs = []
    for index, row in df.iterrows():
        print("LSTM training for PRNG seed {}".format(row['seed']))
        lstm = LatentStateModel(df_row=row)
        lstm.train_recurrent_network()
        lstm.predict_values()
        print("RMSE {}".format(lstm.rmse()))
        RMSEs.append(lstm.rmse())
        line_plot_with_predictions(row=row, data_predicted=lstm.y_predicted)
        cross_correlation(row=row, data_predicted=lstm.y_predicted)
        display_plots()
    print("Average RMSE {}".format(sum(RMSEs)/len(RMSEs)))


if __name__ == '__main__':
    main()
