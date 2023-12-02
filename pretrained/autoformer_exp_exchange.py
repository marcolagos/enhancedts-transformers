from transformers import AutoformerForPrediction
from loader import TimeSeriesDataset, TimeSeriesLoader
import matplotlib.pyplot as plt
import pandas as pd
import os

PATH = "dataset/exchange_rate.csv"
TYPE = "exchange_rate"
DIR = "plots"

def visualization():
    df = pd.read_csv(PATH, parse_dates=['date'])
    df.set_index('date', inplace=True)
    plt.figure(figsize=(10, 5))
    for column in df.columns[:4]:
        plt.plot(df.index, df[column], label=column)
    plt.title('Exchange Rate Trends')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()

    plot_filename = f"visualization_{TYPE}.png"
    plt.savefig(os.path.join(DIR, plot_filename))
    plt.close()

visualization()

dataset = TimeSeriesDataset(PATH)
dataloader = TimeSeriesLoader(dataset, batch_size=870, context_size=840)

while True:
    print("Tensor Batch index: ", dataloader.tensor_batch_index)

    batch = dataloader.next_tensor_batch()

    if batch is None:
        break

    model = AutoformerForPrediction.from_pretrained("elisim/autoformer-exchange-rate-50-epochs")
    # model max lags sequence = 780
    # model context length = 60
    # model prediction length = 30

    # print(model.config)

    outputs = model(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            # static_categorical_features=batch["static_categorical_features"],
            future_values=batch["future_values"],
            future_time_features=batch["future_time_features"],
        )

    loss = outputs.loss
    loss.backward()

    outputs = model.generate(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            # static_categorical_features=batch["static_categorical_features"],
            future_time_features=batch["future_time_features"],
        )

    print("Outputs shape: ", outputs.sequences.shape)       # torch.Size([64, 100, 24])  64 batch size, 100 samples, 24 prediction length
    mean_prediction = outputs.sequences.mean(dim=1)
    print("Outputs mean shape: ", mean_prediction.shape)    # torch.Size([64, 24])  64 batch size, 24 prediction length

    # Plotting the time series forecast future values vs actual future values
    plt.figure(figsize=(10, 5))
    plt.plot(batch["future_values"][0], label="actual")
    plt.plot(mean_prediction[0], label="predicted")
    plt.title('Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()

    plot_filename = f"{TYPE}_batch_{dataloader.tensor_batch_index}.png"
    plt.savefig(os.path.join(DIR, plot_filename))
    plt.close()