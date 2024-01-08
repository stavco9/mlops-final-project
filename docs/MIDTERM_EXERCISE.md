# Project Charter

## Business background

* Who is the client, what business domain the client is in.  
	* The project deals with detecting anomalies in a water circulation system.  
		* Therefore, we assume that:
	* the client's business domain is water utilities industry;
 		* the client is water utility company that provides water to households and businesses in an urban area. 	
* What business problems are we trying to address?
	* As a water utility company, our client's primary business objective is to ensure uninterrupted supply of high-quality water to customers while optimizing the cost-effective maintenance of the water network infrastructure.  
	* Infrastructure maintenance costs are the major operational expense and are in a big part defined by the effectiveness of early detection of signs of infrastructure issues.  
	* Missed anomalies in water circulation have potential to result in water supply failure and substantial expenses in urgent fixing of the problem, repair of the associated damage and more
 	* In contrast, proactive check of potential issues is relatively inexpensive.  
	* Hence, the client's major business challenge is to develop high level of sensitivity to all variety of water supply anomalies, aiming not to overlook any type of issue.

## Scope

To address the business problem at hand, we aim to employ data science tools for detecting anomalies in the water circulation system. Our primary focus is on minimizing undetected issues within the system.

To tackle this challenge, we will use two machine learning models. The first model is based on a convolutional autoencoder, while the second model is based on decision trees. Our objective is to minimize the false negative rate in our system. Failure to identify an anomaly can result in significant financial losses for the client, whereas dispatching a technician is a comparatively cheap solution.

During the model training process, we will identify slices in the data where the models exhibit suboptimal performance. To improve the perfomance in these specific areas, we will generate additional synthetic training samples with similar distribution as the problematic segments. This iterative training approach aims to improve model performance on challenging data slices.

Our tool is each to use and requieres continuous data collection from the water pump, with the user uploading this data to the server.
Upon activation, the application seamlessly processes incoming data, and alerting the designated contact person when an anomaly in the water circulation system is detected.

## Personnel
* Project Developers:
  
	Kiliemah, Stav Cohen, Natalia Meergus, Nitay Cohen

* Project supervisor:

	Dr Ishai Rosenberg

## Metrics
* What are the qualitative objectives? (e.g. reduce user churn)  
  The qualitative objective is to enhance the early detection capabilities of water infrastructure issues, with a specific focus on establishing a proven and balanced ability to detect all kinds of anomalies.  
* What is a quantifiable metric  (e.g. reduce the fraction of users with 4-week inactivity)  
  The quantifiable metric is reduction of False Negative Rate ("Miss rate") of anomalies detection, both average and on key slices.  
* Quantify what improvement in the values of the metrics are useful for the customer scenario (e.g. reduce the  fraction of users with 4-week inactivity by 20%)   
  	* Reduction of average False Negative Rate - by 3%  
	* Reduction of False Negative Rate on a data slice with poorest performance - by 10% 
* What is the baseline (current) value of the metric?
  * Based on LightGBM module, we currently have:
    * Accuracy rate of 92% between the predicted anomaly and the actual one
    * False negative rate of 17% between the predicted anomaly and the actual one
  * Based on Conv_AE module, we currently have:
    * Accuracy rate of 83% between the predicted anomaly and the actual one
    * False negative rate of 35% between the predicted anomaly and the actual one
* How will we measure the metric? (e.g. A/B test on a specified subset for a specified period; or comparison of performance after implementation to baseline)  
  Comparison of False Negative Rates after implementation to baselines' False Negative Rates over the test set.

## Plan
* Phase 1: Dataset exploration
   * Exploring the dataset and determine on which metrics we want to focus on
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
* Phase 4: Choose our algorithms to use in order to achieve the following:
   * Algorithm for generating time-series data based on our dataset (Synthetic data). Based on comparing the test set vs the trained model 
     * Available options are: Time GAN, Deep Echo
   * Algorithm for finding a problematic slices of dataset over the time series (Many false negatives) and tune them. Based on comparing the test set vs the trained model:
     * Available options are: XXXX
* Phase 5: Train our python modules based on our updated dataset which is a result of the execution of our algorithms:
   * lightgbm
   * Conv_AE
* Phase 5: View our results by comparing the test set vs the trained model
   * Check if the following model aspects have been improved by the target improvement we've defined ourselves:
     * false negative rate
     * accuracy rate
* Phase 6: Export the updated dataset to output CSV file to be reusable in the next phases
* Phase 7: Re-run our pipelines with the new generated output

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
<img width="248" alt="image" src="https://github.com/stavco9/mlops-final-project/assets/72156432/24db2cfe-9ec0-4079-8477-14a6a1ba43d8">
</div>
<br>
Types of anomalies. Source: SKAB report


