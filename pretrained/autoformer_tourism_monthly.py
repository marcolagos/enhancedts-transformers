from huggingface_hub import hf_hub_download
import torch
from transformers import AutoformerForPrediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = hf_hub_download(
        repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
    )
batch = torch.load(file)

print(batch["past_values"].shape)                   # torch.Size([64, 61])      64 batch size, 61 context length
print(batch["past_time_features"].shape)            # torch.Size([64, 61, 2])   64 batch size, 61 context length, 2 time features
print(batch["past_observed_mask"].shape)            # torch.Size([64, 61])      64 batch size, 61 context length
print(batch["static_categorical_features"].shape)   # torch.Size([64, 1])       64 batch size, 1 static categorical features
print(batch["future_values"].shape)                 # torch.Size([64, 24])      64 batch size, 24 prediction length
print(batch["future_time_features"].shape)          # torch.Size([64, 24, 2])   64 batch size, 24 prediction length, 2 time features


model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
    )

loss = outputs.loss
loss.backward()

outputs = model.generate(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        future_time_features=batch["future_time_features"],
    )

print(outputs.sequences.shape)  # torch.Size([64, 100, 24])  64 batch size, 100 samples, 24 prediction length
mean_prediction = outputs.sequences.mean(dim=1)
print(mean_prediction.shape)    # torch.Size([64, 24])  64 batch size, 24 prediction length

# Plot the time series forecast future values vs actual future values
plt.figure(figsize=(10, 5))
plt.plot(batch["future_values"][0], label="actual")
plt.plot(mean_prediction[0], label="predicted")
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()