# Exit Report of project:
## Automatic discovery of under-performing data slices in anomaly detection models

## Customer: Randomistan H2O

## Team Members:
* Lecturer: Dr. Ishai Rosenberg
* Students:
  * Nitay Cohen
  * Stav Cohen
  * Kilièmah Ouattara
  * Natalia Meergus

##	Overview

Randomistan H2O, a cost-conscious infrastructure company, uses a proprietary anomaly detection system to preemptively detect its infrastructure failures.  
The performance of the anomaly detection model is critical being in direct relationship with customer expenditures.  
The customer request was to build and deploy an automatic framework for systematic and explainable analysis and improvement of model performance.

Our solution used IBM FreaAI method of automated extraction of data slices.  
It included:
* choice of F2 as a main performance metric instead of currently used accuracy metric
* adaptation of FreaAI method for anomaly detection and chosen metric
* creation of automated report presenting the method results
* execution for the current model and presentation of 2 alternative results to the customer.

The results were well accepted. The customer is positive about proceeding to the next phase model performance improvement based on resulting report.


##	Business Domain
Water utilities industry

##	Business Problem
The customer looks to improve his ability to proactively detect potential failures of the water infrastructure.  
As a first phase, he needs an automatic step to produce a systematic picture of current performance of the model.

##	Data Processing
The original dataset is a set of input files (with a CSV format) of time series data that taken from a water pipe measurement system
* The dataset has the following index:
  * Datetime - Timestamp of when the data was taken (YYYY-MM-DD hh:mm:ss)
* The dataset has the following features:
  * Accelerometer1RMS - The square root of the first vibration acceleration value (g units)
  * Accelerometer2RMS - The square root of the second vibration acceleration value (g units)
  * Current - The amperation of the electric motor (ampere)
  * Pressure - The pressure in the loop after the water bump (bar)
  * Temperature - The temperature of the engine body (Cº)
  * Thermocouple - The temperature of the field in the circulation loop (Cº)
  * Voltage - The voltage of the electric motor (volt)
  * Volume Flow RateRMS - The circulation flow rate of the field inside the loop (Liter / m)
* And the following outputs:
  * Anomaly - Whether the checkpoint is anomolous (0/1)
  * Changepoint - Whether the checkpoint is used for collective anomalies (0/1)

Before executing our models, first we make a dataprocessing in order to make it ready for models training. The general dataprocessing is being done with the following way:
1. Split the dataset into train / valid / test sub-datasets
2. For each sub-dataset, split it into windows and make a smooth curve on each window in order to reduce the residual error
3. Make a data standartization for each window by correcting the scaling difference of each characteristic (For train window we also fit it)

These are the outputs of the data processing:
1. Train / Valid / Test sets of the original data with all the feature's values (X)
2. Windows of each subset (X)
3. Data standartization of each subset (X)
4. Train / Valid / Test sets of the anomalies of the data (Y)

##	Modeling, Validation
* Modeling and validation is described widely in the "Baseline Model" report (For LightGBM) and "Model report" (For FreaAI)
* After we run the models we store the trained model (LightGBM) in Databricks platform with our desired metrics (Outputs of FraeAI model)

##	Solution Architecture
* Overall:
  * Runtime environment: We use our laptops as the runtime environment, with the following installed:
    * OS: MacOS M1 / M2 or Windows 10+ supported
    * Python 3.11 with Anaconda / Miniconda installed
    * The requirements Python packages specified in requirements.txt files installed using pip or conda
    * In a real scenario the runtime environment should be run on an Instance in the cloud
  * Model registry: We use Databricks SaaS platform to register and store our trained models. The Databricks platform is hosted in AWS cloud and we're get authenticated with email + password
* Process:
  * Load the dataset from a set of CSV files and concatinate them into a Dataframe object (Data Processing)
  * Train & Test our baseline model - LightGBM, described in Baseline report
  * Run a FreaAI algorithm based on a single feature of the original dataset and calculate metrics based on true anomalies vs predicted ones (Based on LightGBM) - Described in Model report
  * Register our model and metrics into Databricks platform
  * Load the model from Databricks and make a test using the lightGBM test set in order to validated the model uploaded & downloaded successfuly
* MLFlow:
  * MLFlow is a MLOps library provider by Databricks, a MLOps software company
  * We use MLFlow to register our trained model in the Databricks cloud SaaS platform
  * Using MLFlow, we can handle versions of the trained model
  * In production scenario, we can load the model from the platform and run a prediction against a test set in order to calculate our metrics
 
* Architecture overall diagram:

![alt text](https://github.com/stavco9/mlops-final-project/blob/b40a7fd9c82e4982d1f2d78432054deebd31ef92/docs/MLflowArchitecture.png)

##	Benefits
	
###	Company Benefit (internal only. Double check if you want to share this with your customer)
1. We won our first logo in utilities industry with permission to reference it in our Marketing campaigns and materials.
2. Our innovative adaptation of FreaAI to anomalies detection have proved successful and have good chances to be reused in other projects at lower and more competitive development costs.
3. The specific project is expected to be profitable if the customer will proceed to the Phase 2.


###	Customer Benefit
Phase 1 of the project does not have immediate ROI. Instead, customer achieved:  
* detailed picture or current model performance and its variation
* awareness of cases where current model performance is especially poor
* in-house ability to regularly update the above at no additional cost    

From now on, this up-to-date information will enable the customer to assess feasible performance improvements and to take informed decisions on their further investments in model and infrastructure improvements.


##	Learnings

### 	Project Execution
The customer is the lead from our blog (post about FreaAI). That demonstrates that technical blogs is the effecient way to create initial trust by potential leads and that additional developers should be encouraged to contibute to blogs authoring.   
Involving customers' technical contact persons in metrics discussion probably contributed to their final acceptance of the results.


### Data science / Engineering
We've got many conclusions by analysing our data that can be very useful for the customer, for both baseline model (LightGBM / ConvAE) & main analysis model (FreaAI). The conclusions come into effect with metrics that uplaoded to MLFlow platform and can be used by the customer to find the weaknesses in his pipe system

![alt text](https://github.com/stavco9/mlops-final-project/blob/715f188c122576bf72792673b8a66a0ef7844703/docs/mlflow_screenshot.png)

### Domain
Utilities domain is very costs-oriented. Speak savings.


### Product
FreaAI core algorithm is metric-agnostic, thus allowing easy adaptation to different target metrics, by only changing input and metric calculation on a slice.

###	What's unique about this project, specific challenges
* We chose to analyse the data and find some conclusions about it, rather than improving the model performance
* The data analysis by FraeAI algorithm was not simple because we needed to customize it to use f2 score rather than accuracy
* While using ConvAE model, we needed to make a data processing between the ConvAE and FraeAI, in order to convert the sequence output into a standard numpy vector (For anomalies)

##	Links
* https://github.com/stavco9/mlops-final-project - Source code of the project
* https://dbc-c3108cf4-06da.cloud.databricks.com - Link for the MLFlow Databricks SaaS instance


##	Next Steps
 
Output Report handover on 3/3/2024 by Natalia Meergus  
Phase 2 Planning on 10/3/2024 by Natalia Meergus  
Biweekly check-in status meetings starting 17/3/2024 by Nitay Cohen

## Appendix
* FreaAI documentation: https://towardsdatascience.com/how-to-find-weaknesses-in-your-machine-learning-models-ae8bd18880a3
* MLFlow tutorial: https://www.restack.io/docs/mlflow-knowledge-mlflow-pipeline-tutorial
* Skab dataset: https://drive.google.com/file/d/1dHUevwPp6ftQCEKnRgB4KMp9oLBMSiDM/view
* Skab with LightGBM: https://www.kaggle.com/code/yoshimanaminorisa/anomaly-detection-using-lightgbm#5.-LightGBM-model-for-anomaly-detection
* Skab with ConvAE: https://github.com/waico/SKAB/blob/master/notebooks/Conv_AE.ipynb
