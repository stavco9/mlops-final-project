# Project Charter

## Business background

* Who is the client, what business domain the client is in.  
	The project deals with detecting anomalies in a water circulation system.  
	Thus, we assume that:
	* the business domain is water utilities industry;
 	* the client is a water utility company supplying water to households and businesses in an urban area. 	
* What business problems are we trying to address?  
  	As a water utility company, our primary business objective is uninterrupted supply of high-quality water to customers while cost-optimal maintenance of water network infrastructure.  
	Water infrastructure maintenance costs is a major operational expense and are in a big part defined by level of capability of early detection of signs of infrastructure issues.  
	Missed anomaly in water circulation have potential to result in water supply failure and considerable expenses in urgent fixing of the problem and of the associated damage and other. At the same time, proactive check of potential issues is relatively cheap.  
	Thus, the client's major business problem is staying over-sensitive to all variety of water supply anomalies, in a strive not to miss any kind of them.

## Scope
* What data science solutions are we trying to build?
* What will we do?
* How is it going to be consumed by the customer?

## Personnel
* Who are on this project:
	* Microsoft:
		* Project lead
		* PM
		* Data scientist(s)
		* Account manager
	* Client:
		* Data administrator
		* Business contact
	
## Metrics
* What are the qualitative objectives? (e.g. reduce user churn)  
  The qualitative objective is improving ability of early detection of network issues, with emphasis on proven balanced ability to detect all kinds of them.  
* What is a quantifiable metric  (e.g. reduce the fraction of users with 4-week inactivity)  
  The quantifiable metric is reduction of False Negative Rate ("Miss rate") of anomalies detection, both average and on key slices.  
* Quantify what improvement in the values of the metrics are useful for the customer scenario (e.g. reduce the  fraction of users with 4-week inactivity by 20%)   
  	* Reduction of average False Negative Rate - by 3%  
	* Reduction of False Negative Rate on a data slice with poorest performance - by 10% 
* What is the baseline (current) value of the metric?
  * Based on LightGBM module, we currently have:
    * Accuracy of 0.9257 between the predicted anomaly and the actual one
    * f1 score of 0.9067 between the predicted anomaly and the actual one
  * Based on Conv_AE module, we currently have:
    * Accuracy of 0.786 between the predicted anomaly and the actual one
    * f1 score of 0.77 between the predicted anomaly and the actual one
* How will we measure the metric? (e.g. A/B test on a specified subset for a specified period; or comparison of performance after implementation to baseline)

## Plan
* Phase 1: Dataset exploration
   * Exploring the dataset and determine on which metrics we want to focus on
* Phase 2: Preparing the work environment
   * Create a source code repository
   * Install all python modules on the local machine (Based on requirements.txt)
   * Executing the current lightgbm notebook based on our dataset
   * Executing the current Conv_AE notebook based on our dataset
* Phase 3: Choose our algorithms to use
   * Algorithm for generating time-series data based on our dataset (Synthetic data) - XXXX (Fill in what we've chosen)
* Phase 4: Train our python modules based on our updated dataset:
   * lightgbm
   * Conv_AE
* Phase 5: View our results
   * Check if the accuracy & f1score of the model has been improved
   * For each metric, check if the values of the metric, AFTER using the algorithms in phase 3 has been improved
* Phase 6: Export the updated dataset to output CSV file to be reusable in the next phases

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
