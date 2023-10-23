# Enhanced Time-series Transformers

Enhancements and exploration of Transformer models for long-term time series forecasting, building upon the research presented in the paper "Are Transformers Effective for Time Series Forecasting?"

## Introduction

Recent advancements in Transformer architectures have shown promise in various domains, including natural language processing and computer vision. This repository seeks to explore and potentially improve the application of Transformer models to the specific domain of long-term time series forecasting (LTSF). Stemming from the paper titled "Are Transformers Effective for Time Series Forecasting?", we investigate several modifications tailored to suit the unique challenges of time series data.

## Features

- **Temporal Attention Mechanism**: Designed to give more weight to temporally closer data points.
- **Hybrid Models**: A blend of Transformer's feature extraction and the time-awareness of traditional models.
- **Temporal Embeddings**: Captures the intervals between data points.

## Data

The experiments will be conducted on a variety of time series datasets, ensuring a comprehensive evaluation of our modifications.

## Method

### Temporal Attention Mechanism
By designing an attention mechanism that emphasizes recent data points, we aim to ensure that the model captures short-term patterns and dependencies more effectively.

### Hybrid Models
Merging the capabilities of Transformers and traditional models might provide a balanced approach to handle time series data, capturing both the intricate patterns and temporal structures.

### Temporal Embeddings
Standard positional encodings in Transformers might not fully capture the nuances of time intervals in time series data. By introducing embeddings tailored for this, we aim to provide the model with more contextual information.

## Experimentation

### Model Training
The modified Transformer models will be trained on selected datasets. Consistent training setups will be maintained to ensure a direct comparison between models.

### Performance Evaluation
The forecasting performance of the proposed models will be compared against standard Transformer architectures and the linear models reported in the reference paper.

### Interpretability Analysis
To understand the decision-making process of the models, we'll focus on the distribution of attention weights and feature importance. This will provide insights into how the model processes time series data and the significance of each data point.

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

3. **Unzip the datasets**:
    ```bash
    unzip dataset.zip
    ```

## Citing

If you use this repository for your research or find it useful, please consider citing:

```bibtex
@misc{MarcoLagos2023EnhancedTS,
  title={EnhancedTS-Transformers: Exploring and Enhancing Transformer Architectures for Time Series Forecasting},
  author={Marco Lagos},
  year={2023},
  note={GitHub Repository: https://github.com/marcolagos/enhancedts-transformers}
}
```
