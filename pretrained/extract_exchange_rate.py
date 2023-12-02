"""
    NOT BEING USED
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch

"""
PATH = "dataset/exchange_rate.csv"
df = pd.read_csv(PATH, parse_dates=['date'])
print(df.head())
        date       0       1         2         3         4         5         6      OT
0 1990-01-01  0.7855  1.6110  0.861698  0.634196  0.211242  0.006838  0.525486  0.5930
1 1990-01-02  0.7818  1.6100  0.861104  0.633513  0.211242  0.006863  0.523972  0.5940
2 1990-01-03  0.7867  1.6293  0.861030  0.648508  0.211242  0.006975  0.526316  0.5973
3 1990-01-04  0.7860  1.6370  0.862069  0.650618  0.211242  0.006953  0.523834  0.5970
4 1990-01-05  0.7849  1.6530  0.861995  0.656254  0.211242  0.006940  0.527426  0.5985
"""

class TsfLoader:
    def __init__(self, file_path, context_length, prediction_length):
        self.file_path = file_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.data = self.load_data()

        self.DI = len(self.data)
        self.PI = 0
        self.FI = 0
        self.next_tensor()

    def load_data(self):
        """
        Load data from csv file
        """
        df = pd.read_csv(self.file_path, parse_dates=['date'])
        return df
    
    def next_tensor(self):
        """
        Return next batch of data
        """
        self.PI += self.context_length
        self.FI = self.PI + self.prediction_length

        if self.FI > len(self.data):
            return None
        
        past_values = self.data.iloc[self.PI - self.context_length : self.PI].values
        print(past_values.shape)


        return

    def get_past_values(self):
        """
        past_values:
            (torch.FloatTensor of shape (batch_size, sequence_length)) — Past values of the time series, that serve as context in order to
            predict the future. These values may contain lags, i.e. additional values from the past which are added in order to serve as
            “extra context”. The past_values is what the Transformer encoder gets as input (with optional additional features, such as
            static_categorical_features, static_real_features, past_time_features).
        The sequence length here is equal to context_length + max(config.lags_sequence).

        Missing values need to be replaced with zeros.
        """
        sequence_length = self.context_length
        past_values_list = []
        for i in range(len(self.data) - sequence_length + 1):
            sequence = self.data.iloc[i:i+sequence_length]
            sequence_filled = sequence.fillna(0)
            past_values_list.append(sequence_filled.values)
        past_values_array = np.array(past_values_list)
        past_values_tensor = torch.FloatTensor(past_values_array)
        return past_values_tensor

    def get_past_time_features(self):
        """
        past_time_features 
            (torch.FloatTensor of shape (batch_size, sequence_length, num_features), optional) — Optional time features, which the model
            internally will add to past_values. These could be things like “month of year”, “day of the month”, etc. encoded as vectors
            (for instance as Fourier features). These could also be so-called “age” features, which basically help the model know “at
            which point in life” a time-series is. Age features have small values for distant past time steps and increase 
            monotonically the more we approach the current time step.
        These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where the position encodings
        are learned from scratch internally as parameters of the model, the Time Series Transformer requires to provide additional time 
        features.

        The Autoformer only learns additional embeddings for static_categorical_features.
        """
        sequence_length = self.context_length + self.max_lag
        time_features_list = []

        for i in range(len(self.data) - sequence_length + 1):
            sequence_dates = self.data.index[i:i+sequence_length]
            time_features = self._compute_time_features(sequence_dates)
            time_features_list.append(time_features)
        time_features_array = np.array(time_features_list)
        time_features_tensor = torch.FloatTensor(time_features_array)

        return time_features_tensor
    
    def _compute_time_features(self, dates):
        """
        Compute time features based on the dates.
        :param dates: list of dates
        :return: list of time features
        """
        year = dates.year
        month = dates.month
        day = dates.day
        combined_features = np.column_stack([year, month, day])
        return combined_features

    def get_past_observed_mask(self):
        """
        past_observed_mask (torch.BoolTensor of shape (batch_size, sequence_length), optional) — Boolean mask to indicate which 
        past_values were observed and which were missing. Mask values selected in [0, 1]:
        1 for values that are observed,
        0 for values that are missing (i.e. NaNs that were replaced by zeros).
        """
        sequence_length = self.context_length + self.max_lag
        observed_masks = []

        for i in range(len(self.data) - sequence_length + 1):
            sequence = self.data.iloc[i:i+sequence_length]
            mask = ~sequence.isna()
            observed_masks.append(mask.values)
        observed_masks_array = np.array(observed_masks)
        observed_masks_tensor = torch.BoolTensor(observed_masks_array)
        return observed_masks_tensor

    def get_static_categorical_features(self):
        """
        static_categorical_features 
         (torch.LongTensor of shape (batch_size, number of static categorical features), optional) — Optional
         static categorical features for which the model will learn an embedding, which it will add to the values of the time series.
        Static categorical features are features which have the same value for all time steps (static over time).

        A typical example of a static categorical feature is a time series ID.
        """
        unique_ids = self.data['date'].unique()
        id_to_int = {id: i for i, id in enumerate(unique_ids)}
        mapped_ids = self.data['date'].map(id_to_int).values
        mapped_ids_2d = mapped_ids.reshape(-1, 1)
        static_features = torch.LongTensor(mapped_ids_2d.repeat(self.batch_size, axis=1))
        return static_features

    def get_future_values(self):
        """
        future_values 
            (torch.FloatTensor of shape (batch_size, prediction_length)) — Future values of the time series, that serve as labels for the
            model. The future_values is what the Transformer needs to learn to output, given the past_values.
        See the demo notebook and code snippets for details.

        Missing values need to be replaced with zeros.
        """
        prediction_length = self.prediction_length
        future_values_list = []

        for i in range(len(self.data) - self.context_length - prediction_length + 1):
            future_sequence = self.data.iloc[i + self.context_length : i + self.context_length + prediction_length]
            future_sequence_filled = future_sequence.fillna(0)
            future_values_list.append(future_sequence_filled.values)

        future_values_array = np.array(future_values_list)
        future_values_tensor = torch.FloatTensor(future_values_array)

        return future_values_tensor

    def get_future_time_features(self):
        """
        future_time_features 
            (torch.FloatTensor of shape (batch_size, prediction_length, num_features), optional) — Optional time features, which the mode
            internally will add to future_values. These could be things like “month of year”, “day of the month”, etc. encoded as vector 
            (for instance as Fourier features). These could also be so-called “age” features, which basically help the model know “a which 
            point in life” a time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step
        These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where the position encodings are learned from scratch internally as parameters of the model, the Time Series Transformer requires to provide additional features.

        The Autoformer only learns additional embeddings for static_categorical_features.
        """
        prediction_length = self.prediction_length
        future_time_features_list = []

        for i in range(len(self.data) - self.context_length - prediction_length + 1):
            future_dates = self.data.index[i + self.context_length : i + self.context_length + prediction_length]
            time_features = self._compute_time_features(future_dates)
            future_time_features_list.append(time_features)

        future_time_features_array = np.array(future_time_features_list)
        future_time_features_tensor = torch.FloatTensor(future_time_features_array)

        return future_time_features_tensor


if __name__ == "__main__":
    PATH = "dataset/exchange_rate.csv"
    context_length = 60
    prediction_length = 30
    ts = TsfLoader(PATH, context_length, prediction_length)