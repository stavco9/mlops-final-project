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
    * Accuracy of 0.925704989154013 between the predicted anomaly and the actual one
    * Avarage count of 137 false negative anomaly values which are 0.1706102117061021 of total
  * Based on Conv_AE module, we currently have:
    * Accuracy of 0.8255158973811367 between the predicted anomaly and the actual one
    * Count of 137 false negative anomaly values which are 0.35374971678876216 of total
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
* Phase 3: Train our python modules based on the train slice of our source (input) dataset:
   * lightgbm
   * Conv_AE
* Phase 4: Choose our algorithms to use in order to achieve the following:
   * Algorithm for generating time-series data based on our dataset (Synthetic data). Based on comparing the validate set vs the trained model 
     * Available options are: Time GAN, Deep Echo
   * Algorithm for finding a problematic slices of dataset over the time series (Many false negatives) and tune them. Based on comparing the validate set vs the trained model:
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

Training Data are collected from a water circulation system located in the Skolkovo Institute of Science and Technology (Skoltech). Through an OPC UA communication protocol, data are transferred from the Testbed monitoring system to a MySQL database and then processed to a CSV file. The training data consist of a set of those files obtained from various experimentations (34) describing 7 types of anomalies

Each CSV files contains , 9 times series features representing the state of the system at a given time. Additionally each raw data defining the state is labelled as anomaly or not.


In production, raw data representing the state of the system, will be collected through the same OPC UA protocol, processed and then pass through the binary classification model (as a stream) to detect anomalies at a given time point.
The costumer could decide to check the system base on the model results, either at the time the model detect an anomaly state, or wait for the following stream data point to confirm the previous model output. We expect the costumer to miss less anomalies, by using our model, as we mentioned above.

* Data
  * What data do we expect? Raw data in the customer data sources (e.g. on-prem files, SQL, on-prem Hadoop etc.)
* Data movement from on-prem to Azure using ADF or other data movement tools (Azcopy, EventHub etc.) to move either
  * all the data, 
  * after some pre-aggregation on-prem,
  * Sampled data enough for modeling 

* What tools and data storage/analytics resources will be used in the solution e.g.,
  * ASA for stream aggregation
  * HDI/Hive/R/Python for feature construction, aggregation and sampling
  * AzureML for modeling and web service operationalization
* How will the score or operationalized web service(s) (RRS and/or BES) be consumed in the business workflow of the customer? If applicable, write down pseudo code for the APIs of the web service calls.
  * How will the customer use the model results to make decisions
  * Data movement pipeline in production
  * Make a 1 slide diagram showing the end to end data flow and decision architecture
    * If there is a substantial change in the customer's business workflow, make a before/after diagram showing the data flow.

## Communication
* How will we keep in touch? Weekly meetings?
* Who are the contact persons on both sides?
