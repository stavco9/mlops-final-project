# Project Charter

## Business background

Our client is a player in the water utility sector. 
In this sector it is important to provide efficient and reliable water circulation. 
The water utilities relies on a network of pumps to maintain a consistent and optimal water flow. These pumps are equipped with sensors capable of measuring multiple parameters, such as pressure, flow rate, and temperature, providing crucial insights into the state of the water distribution system.

Our customer's goal is to ensure the uninterrupted and cost-effective delivery of clean water to consumers. This involves managing a network of pumps and monitoring various parameters to maintain operational efficiency, reduce energy consumption, and proactively address issues within the water circulation infrastructure.

Infrastructure maintenance costs are the major operational expense and are in a big part defined by the effectiveness of early detection of signs of infrastructure issues.  

Missed anomalies in water circulation have potential to result in water supply failure and substantial expenses in urgent fixing of the problem, repair of the associated damage and more.
 
In contrast, proactive check of potential issues is relatively inexpensive.  
Hence, the client's major business challenge is to develop high level of sensitivity to all variety of water supply anomalies, aiming not to overlook any type of issue.

## Scope

To address the business problem at hand, we aim to employ data science tools for detecting anomalies in the water circulation system. Our primary focus is on finding "problematic" anomaly slices within the system.

To tackle this challenge, we will use `a machine learning model based on decision trees (FreaAI). Our objective is to find the F2 score of each top-under performing slice of each feature (weighted mean of precision and recall with more weight for recall).` Failure to identify an anomaly can result in significant financial expanses for the client, whereas dispatching a technician is a comparatively inexpensive solution.

During the model execution process, we will identify slices in the data where the models exhibit suboptimal performance `and calculate the F2 score for each one of them. The top under-performed slices and metrics will be uploaded and registered into a cloud platform.`

`Our tool is easy to use and requires continuous data collection from the water infrastructure, with the customer uploading this data to the server.  
Upon activation, the application seamlessly processes incoming data, and alerting the designated contact person when an anomaly in the water circulation system is detected.`

## Personnel
* Project Developers:
  
	Kilièmah Ouattara, Stav Cohen, Natalia Meergus, Nitay Cohen

* Project supervisor:

	Dr Ishai Rosenberg
* Client:  
	Data administrator: Mrs. Agam Flowman  
	Business contact:   Mrs. Marina Drippler

## Metrics
* **Qualitative objectives** 
  The qualitative objective is to enhance the early detection capabilities of water infrastructure issues, with a specific focus on establishing a proven and balanced ability to detect all kinds of anomalies.  
* **Quantifiable metric**  
  The quantifiable metric is `improvement of F2`, both average and on key slices.
It is difficult to estimate the value of detecting undetected anomalies in the system, becuase we don't know the damage which can happen due to a fault in the system. We assume that a valuable improvement `is improvement of general F2 by 1%, and improvement of F2 on a data slice with poorest performance by 5%`. 
* **Metrics' baseline**
  * Based on LightGBM module, we currently have:
    * Accuracy rate of 92% between the predicted anomaly and the actual one
    * False negative rate of 17% between the predicted anomaly and the actual one
    * Recall rate of 83% between the predicted anomaly and the actual one
    * `F2-score of 97.2%`
  * Based on Conv_AE module, we currently have:
    * Accuracy rate of 83% between the predicted anomaly and the actual one
    * False negative rate of 35% between the predicted anomaly and the actual one
    * Recall rate of 65% between the predicted anomaly and the actual one
    * `F2-score of 60%`
* **Metrics measurement**
  * `The top under-performing slice of each feature (Leaf ID)`
  * `The F2 score of each feature after implementation to baselines' prediction anomaly rate versus the actual anomaly rate.`
  * `The F2 improvement rate after removing the top under-performed slice`
  * `The precision for each feature`
  * `The recall for each feature`
  * `The size of the "problematic" slice for each feature (number of measurements)`

## Plan
* Phase 1: Dataset exploration
   * Exploring the dataset to get valuable insights.
* Phase 2: Preparing the work environment
   * Create a source code repository
   * Install all python modules on the local machine (Based on requirements.txt)
   * Executing the current LightGBM notebook based on our dataset
   * Executing the current Conv_AE notebook based on our dataset
* Phase 3: Split our dataset to three slices
  * train
  * test
  * validate 
* Phase 4: Train our python modules based on the train slice of our source (input) dataset and validate it with our validate slice:
   * LightGBM
   * Conv_AE
* `Phase 5: Implement the FreaAI algorithm to find slices of dataset which the model perform badly on and observe the desired metrics.`
* `Phase 6: Upload the trained baseline model & FreaAI metrics to Databricks platform` 
* `Phase 7: Show our results in the Databricks platform to show the FreaAI metrics results (F2, precision, recall), and whether the following has been improved:`
  * `The F2 score improvement for each feature`
* `Phase 8: Load the model from databricks and run a prediction based on the test set`

## Architecture

### 1- Training phase
The client has furnished a collection of datasets aimed at training a model aligned with the business requirements. This compilation comprises a series of CSV files, totaling 34, derived from diverse experimental sources, each delineating 7 distinct types of anomalies (refer to annexes for details). Leveraging this data, we intend to construct a training sample conducive to the development of two models, with a focus on enhancing pertinent metrics. Ultimately, the most optimal model will be deployed into production. The subsequent graph delineates the various stages of the training phase.

<div style="text-align: center;">
<img width="927" alt="image" src="https://github.com/stavco9/mlops-final-project/assets/72156432/f7a796ab-8642-4471-b10a-f62cceaf43a2">
</div>

 

Each dataset encompasses 9 time series features, each reflecting the system's state at a specific moment. Moreover, each raw data point, signifying the system state, is categorized as either an anomaly or not (refer to annexes for specifics).
In addition, the training data will be meticulously balanced. This entails achieving equilibrium among different anomaly types and maintaining a proportional balance between normal states and the overall anomaly states.


### 2- In production
Raw data capturing the system's state will be systematically collected through periodic updates in a CSV file. Subsequently, this data will undergo processing before being streamed through the binary classification model to identify anomalies within a specified timeframe. The customer retains the option to inspect the system based on the model's output. We expect that, the utilization of our model will result in a reduced occurrence of missed anomalies, as previously highlighted.
<br>
<div style="text-align: center;">
<img width="886" alt="image" src="https://github.com/stavco9/mlops-final-project/assets/72156432/08328fee-f1e9-4566-a3e2-e7a48265b500">
</div>
<br>

## Communication

* Meetings:
	* Weekly meetings to check the step by step improvement techniques implementation
 	* Final meeting to discuss the added value of the improvement 
* Contact person:
	* Team: Natalia Meergus
	* Customer: Dr Ishai Rosenberg

## Annexes 
<div style="text-align: center;">
<img width="318" alt="image" src="https://github.com/stavco9/mlops-final-project/assets/72156432/90dba94d-d5d1-48db-9599-047b2a35e30e">
<br>
Dataset features. Source: SKAB report 
</div>
<br><br>

<div style="text-align: center;">
<img width="365" alt="image" src="https://github.com/stavco9/mlops-final-project/assets/72156432/cacba0d4-044f-4e62-bb56-4d1727f6c652">
<br>
Types of anomalies. Source: SKAB report
</div>



