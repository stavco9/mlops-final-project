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

%3CmxGraphModel%3E%3Croot%3E%3CmxCell%20id%3D%220%22%2F%3E%3CmxCell%20id%3D%221%22%20parent%3D%220%22%2F%3E%3CmxCell%20id%3D%222%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3B%22%20edge%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-440%22%20y%3D%22420%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22-330%22%20y%3D%22420.01%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%223%22%20value%3D%22Split%20the%20dataset%20into%20Train%2C%20Validate%20%26amp%3Bamp%3B%20Test%20sets%20-%20Data%20processing%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-330%22%20y%3D%22350%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%224%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BexitX%3D1%3BexitY%3D0.5%3BexitDx%3D0%3BexitDy%3D0%3BentryX%3D0.007%3BentryY%3D0.457%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%223%22%20target%3D%225%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-10%22%20y%3D%22430%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22-40%22%20y%3D%22310%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%225%22%20value%3D%22LightGBM%20training%20model%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-20%22%20y%3D%22360%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%226%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BexitX%3D1.011%3BexitY%3D0.45%3BexitDx%3D0%3BexitDy%3D0%3BexitPerimeter%3D0%3BentryX%3D-0.014%3BentryY%3D0.371%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%225%22%20target%3D%228%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-10%22%20y%3D%22430%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22270%22%20y%3D%22180%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%227%22%20value%3D%22Register%20LightGBM%20model%20%26amp%3Bamp%3B%20FreaAI%20metrics%20into%20MLFlow%20Databricks%20platform%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-30%22%20y%3D%22650%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%228%22%20value%3D%22LightGBM%20test%20model%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22260%22%20y%3D%22370%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%229%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BexitX%3D0.5%3BexitY%3D1%3BexitDx%3D0%3BexitDy%3D0%3BentryX%3D0.564%3BentryY%3D0%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%228%22%20target%3D%2210%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22240%22%20y%3D%22630%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22250%22%20y%3D%22650%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2210%22%20value%3D%22Run%20FreaAI%20based%20on%20the%20trained%20model%20prediction%20vs%20the%20test%20set%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22250%22%20y%3D%22650%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2211%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BexitX%3D-0.021%3BexitY%3D0.607%3BexitDx%3D0%3BexitDy%3D0%3BentryX%3D1.029%3BentryY%3D0.614%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3BexitPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%2210%22%20target%3D%227%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22340%22%20y%3D%22520%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22339%22%20y%3D%22660%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2212%22%20value%3D%22Load%20model%20from%20Model%20to%20run%20predictions%20in%20real%20time%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-310%22%20y%3D%22560%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2213%22%20value%3D%22Show%20desired%20metrics%20in%20MLFlow%20platform%20(F2%20%2B%20Acc)%20based%20on%20FraeAI%20to%20show%20the%20%26quot%3Bproblematic%26quot%3B%20data%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-310%22%20y%3D%22750%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2214%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BentryX%3D1.021%3BentryY%3D0.457%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3B%22%20edge%3D%221%22%20target%3D%2212%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-30%22%20y%3D%22730%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%2280%22%20y%3D%22730.01%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2215%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BentryX%3D1.014%3BentryY%3D0.571%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3BexitX%3D-0.007%3BexitY%3D0.643%3BexitDx%3D0%3BexitDy%3D0%3BexitPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%227%22%20target%3D%2213%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-20%22%20y%3D%22740%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22-157%22%20y%3D%22634%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2216%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BexitX%3D0%3BexitY%3D0.5%3BexitDx%3D0%3BexitDy%3D0%3BentryX%3D0.989%3BentryY%3D0.364%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%2212%22%20target%3D%2220%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-210%22%20y%3D%22680%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22-480%22%20y%3D%22700%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2217%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BexitX%3D-0.039%3BexitY%3D0.429%3BexitDx%3D0%3BexitDy%3D0%3BexitPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%2213%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-304%22%20y%3D%22654%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22-470%22%20y%3D%22720%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2218%22%20value%3D%22Pype%20meaurment%20system%20input%22%20style%3D%22ellipse%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3Brounded%3D1%3BfillColor%3D%23009900%3BstrokeColor%3D%23b85450%3BfontColor%3D%23ffffff%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-760%22%20y%3D%22350%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2219%22%20value%3D%22Load%20input%22%20style%3D%22whiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3BfillColor%3D%230050ef%3BstrokeColor%3D%23001DBC%3BfontColor%3D%23ffffff%3BfontStyle%3D0%3Brounded%3D1%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-550%22%20y%3D%22360%22%20width%3D%22120%22%20height%3D%22120%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2220%22%20value%3D%22Data%20analyst%22%20style%3D%22ellipse%3BwhiteSpace%3Dwrap%3Bhtml%3D1%3Baspect%3Dfixed%3Brounded%3D1%3BfillColor%3D%23009900%3BstrokeColor%3D%23b85450%3BfontColor%3D%23ffffff%3B%22%20vertex%3D%221%22%20parent%3D%221%22%3E%3CmxGeometry%20x%3D%22-610%22%20y%3D%22640%22%20width%3D%22140%22%20height%3D%22140%22%20as%3D%22geometry%22%2F%3E%3C%2FmxCell%3E%3CmxCell%20id%3D%2221%22%20value%3D%22%22%20style%3D%22endArrow%3Dclassic%3Bhtml%3D1%3Brounded%3D1%3BexitX%3D1.011%3BexitY%3D0.557%3BexitDx%3D0%3BexitDy%3D0%3BentryX%3D0.989%3BentryY%3D0.364%3BentryDx%3D0%3BentryDy%3D0%3BentryPerimeter%3D0%3BexitPerimeter%3D0%3B%22%20edge%3D%221%22%20source%3D%2218%22%20parent%3D%221%22%3E%3CmxGeometry%20width%3D%2250%22%20height%3D%2250%22%20relative%3D%221%22%20as%3D%22geometry%22%3E%3CmxPoint%20x%3D%22-388%22%20y%3D%22370%22%20as%3D%22sourcePoint%22%2F%3E%3CmxPoint%20x%3D%22-550%22%20y%3D%22431%22%20as%3D%22targetPoint%22%2F%3E%3C%2FmxGeometry%3E%3C%2FmxCell%3E%3C%2Froot%3E%3C%2FmxGraphModel%3E

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
