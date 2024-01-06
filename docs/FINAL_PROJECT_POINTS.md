**Final project points**



* SKAB Dataset:
    * Developing a benchmark (אמת מידה) that detects anomaly of faults and failures of technical systems
    * A monitoring system that measures the state of the water by time series X axis
    * Detect change points of state of water
* Dynamic time wrapping to predict (לחזות) stock values
    * Distance measure to compare the similarity of two time series, even with different length
    * Used for measure stock price data in different speed over time
    * Identifies similarities between temporary sequences of varying speeds & timings
    * Mainly used in financial domains
    * It finds the average gap between two time serieses
    * It ensures each point in seq A is matches with the most analogous point of seq B
    * It finds the historical data from most similar to less similar
    * Very sensitive to noises, can be resolved by smoothing them out
    * Code:

```
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    def normalize(ts):
    return (ts - ts.min()) / (ts.max() - ts.min())

    def dtw_distance(ts1, ts2):
    ts1_normalized = normalize(ts1)
    ts2_normalized = normalize(ts2)
    distance, _ = fastdtw(ts1_normalized.reshape(-1, 1), ts2_normalized.reshape(-1, 1), dist=euclidean)
    return distance

    def find_most_similar_pattern(n_days):
    current_window = price_data_pct_change[-n_days:].values
    # Adjust to find and store 5 patterns
    min_distances = [(float('inf'), -1) for _ in range(5)]   
    for start_index in range(len(price_data_pct_change) - 2 * n_days - subsequent_days):
        past_window = price_data_pct_change[start_index:start_index + n_days].values
        distance = dtw_distance(current_window, past_window)
        for i, (min_distance, _) in enumerate(min_distances):
            if distance < min_distance:
                min_distances[i] = (distance, start_index)
                break
    return min_distances

    # Get data from yfinance
    ticker = "ASML.AS"
    start_date = '2000-01-01'
    end_date = '2023-07-21'
    data = yf.download(ticker, start=start_date, end=end_date)

    # Transform price data into returns
    price_data = data['Close']
    price_data_pct_change = price_data.pct_change().dropna()

    # Differnt Windows to find patterns on,
    # e.g. if 15, The code will  find the most similar 15 day in the history
    days_to = [15, 20, 30]

    # Number of days for price development observation
    # e.g. if 20, then the subsequent 20 days after pattern window is found will be plotted
    subsequent_days = 20

    for n_days in days_to:
    min_distances = find_most_similar_pattern(n_days)
    fig, axs = plt.subplots(1, 2, figsize=(30, 6))
    axs[0].plot(price_data, color='blue', label='Overall stock price')
    color_cycle = ['red', 'green', 'purple', 'orange', 'cyan']
    subsequent_prices = []

    for i, (_, start_index) in enumerate(min_distances):
        color = color_cycle[i % len(color_cycle)]
        past_window_start_date = price_data.index[start_index]
        past_window_end_date = price_data.index[start_index + n_days + subsequent_days]
        axs[0].plot(price_data[past_window_start_date:past_window_end_date], color=color, label=f"Pattern {i + 1}")
        # Store subsequent prices for median calculation
        subsequent_window = price_data_pct_change[start_index + n_days : start_index + n_days + subsequent_days].values
        subsequent_prices.append(subsequent_window)

    axs[0].set_title(f'{ticker} Stock Price Data')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Stock Price')
    axs[0].legend()

    for i, (_, start_index) in enumerate(min_distances):
        color = color_cycle[i % len(color_cycle)]
        past_window = price_data_pct_change[start_index:start_index + n_days + subsequent_days]
        reindexed_past_window = (past_window + 1).cumprod() * 100
        axs[1].plot(range(n_days + subsequent_days), reindexed_past_window, color=color, linewidth=3 if i == 0 else 1, label=f"Past window {i + 1} (with subsequent {subsequent_days} days)")

    reindexed_current_window = (price_data_pct_change[-n_days:] + 1).cumprod() * 100
    axs[1].plot(range(n_days), reindexed_current_window, color='k', linewidth=3, label="Current window")
    
    # Compute and plot the median subsequent prices
    subsequent_prices = np.array(subsequent_prices)
    median_subsequent_prices = np.median(subsequent_prices, axis=0)
    median_subsequent_prices_cum = (median_subsequent_prices + 1).cumprod() * reindexed_current_window.iloc[-1]
    
    axs[1].plot(range(n_days, n_days + subsequent_days), median_subsequent_prices_cum, color='black', linestyle='dashed', label="Median Subsequent Price Estimation")
    axs[1].set_title(f"Most similar {n_days}-day patterns in {ticker} stock price history (aligned, reindexed)")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Reindexed Price")
    axs[1].legend()

    plt.show()
```


* Time GAN:
    * Time series data generation
    * Generating realistic time series
    * Sequence prediction and time-series representation learning (למידה מייצוג)
    * GAN with control over conditional temporal dynamics afforded by supervised autoregressive models (מודלים בנסיגה אוטומטית מפוקחים)
    * This is the notebook: https://github.com/jsyoon0823/TimeGAN/blob/master/tutorial_timegan.ipynb
* YData profiling:
    * A quick exploratory data analysis on time-series data
    * Used mainly for comparing versions of same dataset
    * Knows to analyse text, files and images
    * https://github.com/ydataai/ydata-profiling/tree/develop

```
import pandas as pd



from ydata_profiling.utils.cache import cache_file

from ydata_profiling import ProfileReport



file_name = cache_file(

   "pollution_us_2000_2016.csv",

   "https://query.data.world/s/mz5ot3l4zrgvldncfgxu34nda45kvb",

)



df = pd.read_csv(file_name, index_col=[0])



# Filtering time-series to profile a single site

site = df[df["Site Num"] == 3003]



# Setting what variables are time series

type_schema = {

   "NO2 Mean": "timeseries",

   "NO2 1st Max Value": "timeseries",

   "NO2 1st Max Hour": "timeseries",

   "NO2 AQI": "timeseries",

}

profile = ProfileReport(

   df,

   tsmode=True,

   type_schema=type_schema,

   sortby="Date Local",

   title="Time-Series EDA for site 3003",

)

profile.to_file("report_timeseries.html")
```

* LGBM:
    * Tree based learning algorithms
    * [https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst)
    * Used in many ML competitions

```
    #fine-tunned hyper paramter
    lgb_params={'objective':'binary',
                'metric':'binary_error',
                'force_row_wise':True,
                'seed':0,
                'learning_rate':0.0424127,
                'min_data_in_leaf':15,
                'max_depth':24,
                'num_leaves':29
            }

    test_acc,test_f1score,test_cm,test_pred,model_lgb=lgb_train_predict(train_x,train_y,valid_x,valid_y,test_x,test_y,params=lgb_params,test_flag=True)

    print('test_acc:' + str(test_acc))
    print('test_f1score:' + str(test_f1score))
    print('test_confusionMatrix')
    display(test_cm)

    plt.figure(figsize=(10,5))
    plt.plot(range(len(test_y)),test_y,linestyle='none', marker='X', color='blue', markersize=5, label='Anomaly')
    plt.plot(range(len(test_pred)),test_pred,linestyle='none', marker='X', color='red', markersize=5, label='Predict',alpha=0.05)
    plt.title('Light GBM')
    plt.xlabel('index')
    plt.ylabel('label')
    plt.legend(loc='best')
```

* Need to improve the accuracy and the f1score
    * (f1 score is like accuracy but balancing precision and recall on the positive class) - That’s the purpose of the final project

* FreaAI - Use mean squared error for linear regression between the predicted vector and the actual one. Detects the drift
* SD Generation: Improve performance of a synthetic data. This is in the second priority because first we need to ensure the rate is improving

* Steps:
    1. Check what metrics we want to improve
        1. Recude the square error. 	E.g. Frea AI
        2. Decreasing false negative values
        3. Changing the loss function we train on to reduce the square error
    2. Analyse the data before training
        1. Detecting problematic slices and improve them
        2. FreaAI -> Need to check the complexability

* Business background:
    * False negative values of anomalies (Leakages of the pipeline) costs a lot of money

* Metrics:
    * Anomaly by time series on the test test (Proposed value vs actual value)
    * False negative over time (See it’s reducing)
    * Mean square error rate (Not for final metrics, but for us)

* Data architecture:
    * Read and write CSV (Mayber read & write to s3 bucket) ?

* Tasks - for 9/1:
    * Writing the business background, metrics (explain as in document, needs a little research), architecture and customers
    * Plan:
        * Exploring the data generation links and choose the best models to use
    * Business background - Natalie
    * Scope - Nitai
    * Metrics - desired state - Natalie
    * Metrics - run the notebook - Stav
    * Architecture - Killàme
    * Plan - Stav (can start)
    * Communication - Killàme
