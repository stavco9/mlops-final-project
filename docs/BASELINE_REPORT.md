# Baseline Model Report

_Baseline model is the the model a data scientist would train and evaluate quickly after he/she has the first (preliminary) feature set ready for the machine learning modeling. Through building the baseline model, the data scientist can have a quick assessment of the feasibility of the machine learning task._

When applicable, the Automated Modeling and Reporting utility developed by TDSP team of Microsoft is employed to build the baseline models quickly. The baseline model report is generated from this utility easily. 

> If using the Automated Modeling and Reporting tool, most of the sections below will be generated automatically from this tool. 

## Analytic Approach
* What is target definition
* What are inputs (description)
* What kind of model was built?

## Model Description

Before executing our FraeAI Decision tree described in the model report, we first train the following model in order to achieve a comprasion of predicted vs actual anomalies taht will be used in our main model. Our model is LightGBM
* We use LightGBM model to predict anomalies based on our features (Described in the final report), but with the current modifications:
  * We use our data standartization output from the data pre-processing as input
  * We split each one of the 8 features into 3 (Total of 24 features):
    * Min value whithin each window
    * Mean value whithin each window
    * Max value whithin each window
* We train the LightGBM model based on the new version of the dataset (24 features) where we use the train set as the train input, and a combination of the train set with the valid set as the valid input
* The train model hyperparameters are:
  * learning_rate - A random value
  * min_data_in_leaf - A random value
  * max_depth - A random value
  * num_leaves - A random value
* The main outputs of the train are:
  * Trained model
  * Train accuracy (Calculated with sklearn library)
  * Valid accuracy (Calculated with sklearn library)
* Then we run a train & prediction on the LightGBM model based on the test set features, and we predict the anomalies as the follow:
  * x < 0.5 -> 0
  * x >= 0.5 -> 1
* The test model hyperparameters are:
  * learning_rate - 0.0424127
  * min_data_in_leaf - 15
  * max_depth - 24
  * num_leaves - 29
* The main outputs of the tests are:
  * Test accuracy (Calculated with sklearn library)
  * Predicted outputs of anomalies
  * "True" outputs of anomalies (of the test set)

## Results (Model Performance)
* Train accuracy: 0.66
* valid acc: 0.66
* Test accuracy: 0.9257

* This is the true anomaly vs prediction graph
  
![image](https://github.com/stavco9/mlops-final-project/assets/33497599/1bfe4139-b822-4e72-9bc0-f476d356b7c0)
* This is the lightGMB feature importance graph

![image](https://github.com/stavco9/mlops-final-project/assets/33497599/2e1e7526-8b82-465d-a3df-3264dd1716ea)


## Model Understanding

* Variable Importance (significance)

* Insight Derived from the Model



## Conclusion and Discussions for Next Steps

* Conclusion on Feasibility Assessment of the Machine Learning Task

* Discussion on Overfitting (If Applicable)

* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling
