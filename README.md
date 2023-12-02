# Enhanced Time-series Transformers

Enhancements and exploration of Transformer models for long-term time series forecasting, building upon the research presented in the paper "Are Transformers Effective for Time Series Forecasting?"

## Introduction

Recent advancements in Transformer architectures have shown promise in various domains, including natural language processing and computer vision. This repository seeks to explore and potentially improve the application of Transformer models to the specific domain of long-term time series forecasting (LTSF). Stemming from the paper titled "Are Transformers Effective for Time Series Forecasting?", we investigate several modifications tailored to suit the unique challenges of time series data.

## Setting Up the Development Environment

### Prerequisites:

-   Ensure you have Python 3.x installed. You can check your version with:
    ```bash
    python --version
    ```

### Steps:

1. **Clone the Repository**

    First, clone the repository to your local machine:

    ```bash
    git clone https://github.com/marcolagos/enhancedts-transformers.git
    cd enhancedts-transformers
    ```

2. **Set Up a Virtual Environment**

    It's recommended to use a virtual environment to ensure dependencies are isolated from your system Python. Create a new virtual environment using the `venv` module:

    ```bash
    python -m venv tts_env
    ```

    This command creates a virtual environment named `env` in your project directory.

3. **Activate the Virtual Environment**

    Before installing dependencies or running the project, you need to activate the virtual environment:

    - **On macOS and Linux**:

        ```bash
        source tts_env/bin/activate
        ```

    - **On Windows** (Command Prompt):

        ```bash
        .\tts_env\Scripts\activate
        ```

    - **On Windows** (PowerShell):
        ```bash
        .\tts_env\Scripts\Activate.ps1
        ```

    After activation, your terminal prompt should change to show the name of the activated environment (`tts_env`).

4. **Install Dependencies**

    With the virtual environment activated, install the project's dependencies:

    ```bash
    pip3 install -r requirements.txt
    ```

5. **Deactivate the Virtual Environment**

    Once you're done working on the project, you can deactivate the virtual environment to return to your system's Python:

    ```bash
    deactivate
    ```

## Running Tests

### Downlaod datasets

First download the datasets from the following google drive: https://drive.google.com/drive/u/0/folders/1NdU7D7y1VdQN_tsYTNHby5oDdp3L6O6v and place it in a top-level directory named `./dataset/`. In addition, to run the Hugging-Face pretrained model scripts, place another datasets folder in `./pretrained/dataset/`.

### Scripts

Navigate to `./scripts/EXP-LongForecasting` folder. Run any of the scripts in that directory with any configuration desired. Keep in mind that the scripts for running the Autoformer and DLinear models require a GPU, while the hugging-face autoformer tests recreation under `hf_script.ipynb` requires x86 architecture chips (no M1 chip). The main scripts used in this analysis are:
```
./Stat_Long.sh
./Linear-I.sh
hf_script.ipynb
Formers_Long.sh
./Linear/*
```

In addition, the `./pretrained/` folder holds attempts at the application of pre-trained autoformer models in the scripts:
```
autoformer_exp_electricity.py
autoformer_exp_exchange.py
autoformer_tourism_monthly.py
```
