# Exit Report of project:
## Automatic discovery of under-performing data slices in anomaly detection models

## Customer: <Enter Customer Name\>

## Team Members:
* Lecturer: Dr. Ishai Rosenberg
* Students:
  * Nitay Cohen
  * Stav Cohen
  * Kilièmah Ouattara
  * Natalia Meergus

##	Overview

<Executive summary of entire solution, brief non-technical overview\>

##	Business Domain
<Industry, business domain of customer\>

##	Business Problem
<Business problem and exact use case(s), why it matters\>

##	Data Processing
The original dataset is a set of input files (with a CSV format) of time series data that taken from a water pipe measurement system
The dataset has the following index:
* Datetime - Timestamp of when the data was taken (YYYY-MM-DD hh:mm:ss)
The dataset has the following features:
* Accelerometer1RMS - The square root of the first vibration acceleration value (g units)
* Accelerometer2RMS - The square root of the second vibration acceleration value (g units)
* Current - The amperation of the electric motor (ampere)
* Pressure - The pressure in the loop after the water bump (bar)
* Temperature - The temperature of the engine body (Cº)
* Thermocouple - The temperature of the field in the circulation loop (Cº)
* Voltage - The voltage of the electric motor (volt)
* Volume Flow RateRMS - The circulation flow rate of the field inside the loop (Liter / m)
And the following outputs:
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
Before executing our FraeAI Decision tree described in the model report, we first train one of the two following models in order to achieve a comprasion of predicted vs actual anomalies taht will be used in our main model
* LightGBM
  * We use LightGBM model to predict anomalies based on our features, but with the current modifications:
    * We use our data standartization output from the data pre-processing as input
    * We split each one of the 8 features into 3 (Total of 24 features):
      * Min value whithin each window
      * Mean value whithin each window
      * Max value whithin each window
    * We train the LightGBM model based on the new version of the dataset (24 features) where we use the train set as the train input, and a combination of the train set with the valid set as the valid input
    * The main outputs of the train are:
      * Trained model
      * Train accuracy
      * Valid accuracy
    * Then we run a prediction on the trained LightGBM model based on the test set features, and we predict the anomalies as the follow:
      * x < 0.5 -> 0
      * x >= 0.5 -> 1
    * The main outputs of the tests are:
      * Test accuracy
      * Predicted outputs of anomalies
      * "True" outputs of anomalies (of the test set)

##	Solution Architecture
<Architecture of the solution, describe clearly whether this was actually implemented or a proposed architecture. Include diagram and relevant details for reproducing similar architecture. Include details of why this architecture was chosen versus other architectures that were considered, if relevant\>

##	Benefits
	
###	Company Benefit (internal only. Double check if you want to share this with your customer)
<What did our company gain from this engagement? ROI, revenue,  etc\>

###	Customer Benefit
What is the benefit (ROI, savings, productivity gains etc)  for the customer? If just POC, what is estimated ROI? If exact metrics are not available, why does it have impact for the customer?\>

##	Learnings

### 	Project Execution
<Learnings around the customer engagement process\>

### Data science / Engineering
<Learnings related to data science/engineering, tips/tricks, etc\>


### Domain
<Learnings around the business domain, \>


### Product
<Learnings around the products and services utilized in the solution \>

###	What's unique about this project, specific challenges
<Specific issues or setup, unique things, specific challenges that had to be addressed during the engagement and how that was accomplished\>

##	Links
* https://github.com/stavco9/mlops-final-project - Source code of the project
* https://dbc-c3108cf4-06da.cloud.databricks.com - Link for the MLFlow Databricks SaaS instance


##	Next Steps
 
<Next steps. These should include milestones for follow-ups and who 'owns' this action. E.g. Post- Proof of Concept check-in on status on 12/1/2016 by X, monthly check-in meeting by Y, etc.\>

## Appendix
<Other material that seems relevant – try to keep non-appendix to <20 pages but more details can be included in appendix if needed\>
