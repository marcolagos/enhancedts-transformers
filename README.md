# Enhanced Time-series Transformers

An exploration and enhancement of Transformer models specifically tailored for time series forecasting.

## Overview

This repository aims to delve deeper into the applicability of Transformer architectures for long-term time series forecasting (LTSF). Stemming from the research presented in the paper "Are Transformers Effective for Time Series Forecasting?", we seek to improve upon existing models by introducing several modifications to better suit the unique challenges posed by time series data.

## Features

- **Temporal Attention Mechanism**: An attention mechanism that gives more weight to temporally closer data points.
- **Hybrid Models**: Combining the feature extraction prowess of Transformers with the temporal structure preservation capabilities of traditional models.
- **Temporal Embeddings**: Embeddings designed to capture the temporal intervals between data points.

## Getting Started

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/marcolagos/enhancedts-transformers.git
    cd enhancedts-transformers
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Main Script**:
    ```bash
    python main.py
    ```

## Datasets

The models are trained and validated on several time series datasets, which will be detailed here.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
