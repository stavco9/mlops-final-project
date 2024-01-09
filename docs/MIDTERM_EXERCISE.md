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

To address the business problem at hand, we aim to employ data science tools for detecting anomalies in the water circulation system. Our primary focus is on minimizing undetected issues within the system.

To tackle this challenge, we will use two machine learning models. The first model is based on a convolutional autoencoder, while the second model is based on decision trees. Our objective is to minimize the false negative rate in our system. Failure to identify an anomaly can result in significant financial expanses for the client, whereas dispatching a technician is a comparatively inexpensive solution.

During the model training process, we will identify slices in the data where the models exhibit suboptimal performance. To improve the perfomance in these specific areas, we will generate additional synthetic training samples with similar distribution as the problematic segments. This iterative training approach aims to improve model performance on challenging data slices.

Our tool is easy to use and requieres continuous data collection from the water pump, with the customer uploading this data to the server.
Upon activation, the application seamlessly processes incoming data, and alerting the designated contact person when an anomaly in the water circulation system is detected.

## Personnel
* Project Developers:
  
	Kiliemah, Stav Cohen, Natalia Meergus, Nitay Cohen

* Project supervisor:

	Dr Ishai Rosenberg

## Metrics
* **Qualitative objectives** 
  The qualitative objective is to enhance the early detection capabilities of water infrastructure issues, with a specific focus on establishing a proven and balanced ability to detect all kinds of anomalies.  
* **Quantifiable metric**  
  The quantifiable metric is reduction of False Negative Rate ("Miss rate") of anomalies detection, both average and on key slices.
It is difficult to estimate the value of detecting undetected anomalies in the system, becuase we don't know the damage which can happen due to a fault in the system. We assume that a valuable improvments are reduction of average False Negative Rate (and increase the recall rate) by 3%, and reduction of False Negative Rate on a data slice with poorest performance (and increase the recall rate) by 10%. 
* **Metrics' basline**
  * Based on LightGBM module, we currently have:
    * Accuracy rate of 92% between the predicted anomaly and the actual one
    * False negative rate of 17% between the predicted anomaly and the actual one
    * Recall rate of 83% between the predicted anomaly and the actual one
  * Based on Conv_AE module, we currently have:
    * Accuracy rate of 83% between the predicted anomaly and the actual one
    * False negative rate of 35% between the predicted anomaly and the actual one
    * Recall rate of 65% between the predicted anomaly and the actual one
* **Metrics measurment**  
  Comparison of False Negative rates and recall after implementation to baselines' False Negative rates and recall over the test set.

## Plan
* Phase 1: Dataset exploration
   * Exploring the dataset to get valuable insights.
* Phase 2: Preparing the work environment
   * Create a source code repository
   * Install all python modules on the local machine (Based on requirements.txt)
   * Executing the current lightgbm notebook based on our dataset
   * Executing the current Conv_AE notebook based on our dataset
* Phase 3: Split our dataset to three slices
  * train
  * test
  * validate 
* Phase 3: Train our python modules based on the train slice of our source (input) dataset and validate it with our validate slice:
   * lightgbm
   * Conv_AE
* Phase 4: Implement an algorithm to find slices of dataset which the model perform badbly on. That means low recall rate.
* Phase 5: Choose an algorithm for generating time-series data (Synthetic data) with similar distribution as the problematic slices. Possible packages are Time GAN and Deep Echo.
* Phase 6: Use the extended data sets to train again and fine tune the models:
   * lightgbm
   * Conv_AE
* Phase 7: Show our results by comparing the perfomance on the test set with and without our method. We will apply same algorithm for detecting problematic slices so get customer will be aware in case that our model has sub optimal performance over some data.
Check if the following model aspects have been improved by the target improvement we've defined ourselves:
     * False negative rate
     * Recall rate
* Phase 8: Export the trained model and create an application which wraps the model with simlpe user interface.
* 
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

* Meetings :
	* Weekly meetings to check the step by step improvement techniques implementation
 	* Final meeting to discuss the added value of the improvement 
* Contact person :
	* Team : Natalia Meergus
	* Customer : Dr Ishai Rosenberg

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



