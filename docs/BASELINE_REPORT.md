# Baseline Model Report


## Analytic Approach
The primary objective of the baseline LightGBM model is to predict anomalous states within a water distribution system. Utilizing sequential data derived from various features, the model identifies points of anomaly. The table below outlines these features.<br>
￼
![Pasted Graphic](https://github.com/stavco9/mlops-final-project/assets/72156432/d18e6b40-6a2c-4f9b-b982-f4b1c0d883a5)

The target variable is binary, indicating whether the current state is anomalous or not.
LightGBM stands as a gradient boosting framework employing tree-based learning algorithms. Engineered for distributed and efficient operation, it offers several advantages:
* Accelerated training speed and heightened efficiency.
* Reduced memory usage.
* Enhanced accuracy.
* Support for parallel, distributed, and GPU learning.
* Capability to handle large-scale datasets.


## Model Description

Prior to being fed into the model, the features undergo preprocessing through an initial step. This process aims to derive sub-features from sequential data.
Initially, a Kaiser window is applied to smooth the sequences, thereby mitigating residual errors. Subsequently, a scaler is applied to standardize all features. Finally, sub-features are extracted from them. For each feature, the mean, minimum, and maximum values are computed within windows consisting of 10 timestamps.

At the conclusion of the preprocessing step, 24 features are extracted from various feature windows. These sub-features serve as the primary input for the model.
Following this, a fine-tuning step is executed on the training data to determine the optimal model and learning parameters, which are as follows:

* objective: binary 
* metric: binary_error
* force_row_wise: True
* seed: 0
* learning_rate: 0.0424127
* min_data_in_leaf: 15
* max_depth: 24
* num_leaves: 29

## Results (Model Performance)
* Test Accuracy : 0.919
* Recall : 1
* Precision : 0.876
* F1-score : 0.898
* F2-score : 0.972

This is the true anomaly vs prediction graph

![image](https://github.com/stavco9/mlops-final-project/assets/33497599/1bfe4139-b822-4e72-9bc0-f476d356b7c0)


## Model Understanding

The LightGBM model facilitates comprehension of prediction mechanisms owing to its tree-based methodology. Through this approach, we can discern which features effectively partition the training data, as indicated by impurity metrics. The graph provided illustrates the importance of features derived from the baseline model. Notably, it reveals that the minimum volume flow value across timestamp windows emerges as the most significant feature.<br>
![image](https://github.com/stavco9/mlops-final-project/assets/33497599/2e1e7526-8b82-465d-a3df-3264dd1716ea)<br>
![Anomaly -](https://github.com/stavco9/mlops-final-project/assets/72156432/5d3dd7e5-e57a-41c5-a072-cd16904dc00f)
Also from the confusion matrix, the baseline model tends to generate false alarm significantly, impacting the precision.


## Conclusion and Discussions for Next Steps

The baseline model demonstrates acceptable accuracy; however, it exhibits a lower F1 score due to a notably high false alarm rate, consequently reducing precision and overall F1 score. From a business perspective, it's imperative for a model to prioritize recall while also enhancing precision. In the next phase, our focus will be on identifying strategies to improve the F2 metric. This involves detecting specific data segments that have the potential to enhance model performance through refined training techniques.
