### MLOps final project based on SKAB dataset

Prerequirements:
```
1. Install Anaconda / Miniconda in your machine (Mandatory for MacOS), optional for other OS
2. Python 3.11.x installed
3. Run the following command (Be sure you install the up-to-date packages. If you get an error about some package while running, run pip install <package> --upgrade):
    conda install --yes --file requirements.txt # For MacOS
    pip install -r requirements.txt # For Windows
4. Set the following environment variables (In Windows, you can set the env vars using this guide https://phoenixnap.com/kb/windows-set-environment-variable):
     export MLFLOW_TRACKING_USERNAME=your_email@gmail.com # For MacOS
     export MLFLOW_TRACKING_PASSWORD=******** # For MacOS
     set MLFLOW_TRACKING_USERNAME=your_email@gmail.com # For Windows
     set MLFLOW_TRACKING_PASSWORD=******** # For Windows
```
Executing LightGBM (Without MLFlow):
```
python main.py --model lightgbm
```
Executing LightGBM (With MLFlow):
```
python main.py --model lightgbm --run-mlflow
```
Executing ConvAE (Without MLFlow):
```
python main.py --model convae
```
Executing ConvAE (With MLFlow):
```
python main.py --model convae --run-mlflow
```

* One important note:
  * Unfortunately, I failed to execute the MLFlow part on Windows, so if you run it on Windows please run it without the "--run-mlflow" flag. Rather, I can show you the process from my Laptop using a share screen on a Zoom meeting