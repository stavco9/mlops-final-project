### MLOps final project based on SKAB dataset

Prerequirements:
```
1. Install Anaconda / Miniconda in your machine (Mandatory for MacOS), optional for other OS
2. Python 3.11.x installed
3. Run the following command:
    conda install --yes --file requirements.txt # For MacOS
    pip3 install -r requirements.txt # For other OS
4. Set the following environment variables:
     export MLFLOW_TRACKING_USERNAME=your_email@gmail.com
     export MLFLOW_TRACKING_PASSWORD=********
```
Executing LightGBM:
```
python3 main.py --model lightgbm
```
Executing ConvAE:
```
python3 main.py --model convae
```