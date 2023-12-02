import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TimeSeriesDataset():
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.time_data = pd.DataFrame()
        self.time_data['year'] = self.data['date'].dt.year
        self.time_data['month'] = self.data['date'].dt.month
        self.time_data['day'] = self.data['date'].dt.day
        self.time_data['dayofweek'] = self.data['date'].dt.dayofweek
        self.data.drop('date', axis=1, inplace=True)

    def __len__(self):
        return len(self.data)
    
class TimeSeriesLoader():
    def __init__(self, dataset, batch_size=100, context_size=90):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size

        self.num_batches = len(self.dataset) // batch_size
        self.num_tensor_batches = len(self.dataset.data.columns)
        self.tensor_batch_index = 0

        self.context_val = context_size
        self.pred_val = batch_size - self.context_val

        # print("Context value: ", context_val)
        # print("Predication value: ", pred_val)

    def next_tensor_batch(self):
        """
        Return next batch of data
        """
        if self.tensor_batch_index >= self.num_tensor_batches:
            return None
        
        curr_data_col = self.dataset.data.columns[self.tensor_batch_index]
        curr_data = self.dataset.data[curr_data_col]

        context_val = self.context_val
        pred_val = self.pred_val

        # past_values with dimension (batch_size, batch_size * context_size)
        past_values = torch.zeros((self.batch_size, context_val))
        # print("Past values shape: ", past_values.shape)

        # past_time_features with dimension (batch_size, batch_size * context_size, time_data.shape[1])
        past_time_features = torch.zeros((self.batch_size, context_val, self.dataset.time_data.shape[1]))
        # print("Past time features shape: ", past_time_features.shape)

        # past_observed_mask with dimension (batch_size, batch_size * context_size)
        past_observed_mask = torch.zeros((self.batch_size, context_val))
        # print("Past observed mask shape: ", past_observed_mask.shape)

        # static_categorical_features with dimension (batch_size, 1)
        static_categorical_features = torch.zeros((self.batch_size, 1)) # index
        # print("Static categorical features shape: ", static_categorical_features.shape)

        # future_values with dimension (batch_size, batch_size * (1 - context_size))
        future_values = torch.zeros((self.batch_size, pred_val))
        # print("Future values shape: ", future_values.shape)

        # future_time_features with dimension (batch_size, batch_size * (1 - context_size), time_data.shape[1])
        future_time_features = torch.zeros((self.batch_size, pred_val, self.dataset.time_data.shape[1]))
        # print("Future time features shape: ", future_time_features.shape)
        
        for i, batch_start in enumerate(range(0, self.num_batches, self.batch_size)):
            batch_end = batch_start + self.batch_size

            # print("Batch index: ", i)
            # print("Batch start: ", batch_start)
            # print("Batch end: ", batch_end)

            batch_data = curr_data[batch_start:batch_end]
            batch_time_data = self.dataset.time_data[batch_start:batch_end]

            # print("Batch data shape: ", batch_data.shape)
            # print("Batch time data shape: ", batch_time_data.shape)

            # past_values
            past_values[i] = torch.FloatTensor(batch_data[:context_val])
            # past_time_features
            past_time_features[i] = torch.FloatTensor(batch_time_data[:context_val].values)
            # past_observed_mask
            past_observed_mask[i] = torch.FloatTensor(batch_data[:context_val].values)
            # static_categorical_features
            static_categorical_features[i] = torch.FloatTensor([i])
            # future_values
            future_values[i] = torch.FloatTensor(batch_data[context_val:].values)
            # future_time_features
            future_time_features[i] = torch.FloatTensor(batch_time_data[context_val:].values)

        self.tensor_batch_index += 1

        print(past_values.shape)
        print(past_time_features.shape)
        print(past_observed_mask.shape)
        print(static_categorical_features.shape)
        print(future_values.shape)
        print(future_time_features.shape)

        return { 
            "past_values": past_values,
            "past_time_features": past_time_features,
            "past_observed_mask": past_observed_mask,
            "static_categorical_features": static_categorical_features,
            "future_values": future_values,
            "future_time_features": future_time_features,
        }

# file_name = 'dataset/exchange_rate.csv'
# dataset = TimeSeriesDataset(file_name)
# dataloader = TimeSeriesLoader(dataset)

# while True:
#     batch = dataloader.next_tensor_batch()
#     if batch is None:
#         break