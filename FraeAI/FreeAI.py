# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# libraries importing
import pandas as pd
import matplotlib.pyplot as plt

# additional modules
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#import graphviz
#import dtreeviz.trees
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
import itertools
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import plot_tree
from IPython.display import display


def CreateModels(test_data , accurate_prediction, number_of_features, max_models_number = float('inf')):
  #list of all the returned modules
  models_res = []
  categories_list = list(test_data)
  num_of_models = 0
  models_order =  list(itertools.combinations(categories_list, number_of_features))
  #for categories in itertools.combinations(set(categories_list), number_of_features):
  for categories in models_order:
    if num_of_models > max_models_number:
      break
    num_of_models += 1

    if number_of_features == 1:
      f_names = [categories[0]]
    else:
      f_names = [categories[0], categories[1]]

    #Featurs values
    X = test_data[f_names]
    #whether we predict correctly for this input
    y = accurate_prediction

    #create Decision tree
    model = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=1)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    #plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
    #plot_tree(model, filled=True, feature_names=f_names, node_ids=True)
    #plt.savefig(f'{f_names}' + '.png')
    #plt.clf()

    models_res.append(model)

  return models_res, models_order

def create_ranges_2_features_rec(tree, node_id, min_val_f1, max_val_f1, min_val_f2, max_val_f2):
  res =[]
  left_child_index = tree.children_left[node_id]
  right_child_index = tree.children_right[node_id]
  if left_child_index!=-1:
    if tree.feature[node_id] == 0:#feature1 is the split
      res.append([left_child_index,min_val_f1, tree.threshold[node_id], min_val_f2, max_val_f2])
      res = res + create_ranges_2_features_rec(tree,left_child_index, min_val_f1, tree.threshold[node_id], min_val_f2, max_val_f2)
    else:#feature2 is the split
      res.append([left_child_index,min_val_f1, max_val_f1, min_val_f2, tree.threshold[node_id]])
      res = res + create_ranges_2_features_rec(tree,left_child_index, min_val_f1, max_val_f1,min_val_f2, tree.threshold[node_id])
  if right_child_index!=-1:
    if tree.feature[node_id] == 0:#feature1 is the split
      res.append([right_child_index,tree.threshold[node_id], max_val_f1, min_val_f2, max_val_f2])
      res = res + create_ranges_2_features_rec(tree,right_child_index, tree.threshold[node_id],max_val_f1, min_val_f2, max_val_f2)
    else:
      res.append([right_child_index, min_val_f1, max_val_f1, tree.threshold[node_id], max_val_f2])
      res = res + create_ranges_2_features_rec(tree, right_child_index, min_val_f1, max_val_f1, tree.threshold[node_id], max_val_f2)
  return res
def create_ranges_rec(tree, node_id, min_val, max_val):
  res = []
  left_child_index = tree.children_left[node_id]
  right_child_index = tree.children_right[node_id]
  if left_child_index!=-1:
    res.append([left_child_index,min_val, tree.threshold[node_id]])
    res = res + create_ranges_rec(tree,left_child_index, min_val, tree.threshold[node_id])
  if right_child_index!=-1:
    res.append([right_child_index,tree.threshold[node_id], max_val])
    res =res  +create_ranges_rec(tree,right_child_index, tree.threshold[node_id],max_val)
  return res

def create_ranges(tree):
  res =[]
  min_val = -1000
  max_val = float('inf')
  left_child_index = tree.children_left[0]
  right_child_index = tree.children_right[0]
  if left_child_index!=-1:
    res.append([left_child_index,min_val, tree.threshold[0]])
    res = res + create_ranges_rec(tree,left_child_index, min_val, tree.threshold[0])
  if right_child_index!=-1:
    res.append([right_child_index,tree.threshold[0], max_val])
    res = res + create_ranges_rec(tree,right_child_index, tree.threshold[0],max_val)
  return res


def create_ranges_2_features(tree):
  res =[]
  min_val_f1= -1000
  max_val_f1 = float('inf')
  min_val_f2 = -1000
  max_val_f2 = float('inf')
  left_child_index = tree.children_left[0]
  right_child_index = tree.children_right[0]
  if left_child_index!=-1:
    if tree.feature[0] == 0:#feature1 is the split
      res.append([left_child_index,min_val_f1, tree.threshold[0], min_val_f2, max_val_f2])
      res = res + create_ranges_2_features_rec(tree,left_child_index, min_val_f1, tree.threshold[0], min_val_f2, max_val_f2)
    else:#feature2 is the split
      res.append([left_child_index,min_val_f1, max_val_f1, min_val_f2, tree.threshold[0]])
      res = res + create_ranges_2_features_rec(tree,left_child_index, min_val_f1, max_val_f1,min_val_f2, tree.threshold[0])
  if right_child_index!=-1:
    if tree.feature[0] == 0:#feature1 is the split
      res.append([right_child_index,tree.threshold[0], max_val_f1, min_val_f2, max_val_f2])
      res = res + create_ranges_2_features_rec(tree,right_child_index, tree.threshold[0],max_val_f1, min_val_f2, max_val_f2)
    else:
      res.append([right_child_index, min_val_f1, max_val_f1, tree.threshold[0], max_val_f2])
      res = res + create_ranges_2_features_rec(tree, right_child_index, min_val_f1, max_val_f1, tree.threshold[0], max_val_f2)
  return res
def saveRelevantInfo2_no_acc(model,test_pred1, true_pred, orifinal_data, data, original_acc, acc_deg, min_sample_number, max_impurity, number_of_features = 1, Metric = 'f2'):
  res = []
  test_pred= np.squeeze(test_pred1)

  m = model
  tree = m.tree_
  #single feature
  max_index = data.index.max()
  max_index_original = orifinal_data.index.max()+1
  all_indexes = set(range(max_index + 1))
  original_f2 = fbeta_score(true_pred, test_pred, average='binary', beta=2)
  original_recall = recall_score(true_pred, test_pred)
  if number_of_features == 1:
    ranges = sorted(create_ranges(tree), key=lambda x: x[0])
    for node_id in range(len(ranges)):
      node_min = ranges[node_id][1]
      node_max = ranges[node_id][2]
      correct_indices = data[(data[data.columns[0]].between(node_min, node_max))].index
      size = len(orifinal_data[(orifinal_data[data.columns[0]].between(node_min, node_max))].index)
      non_slice_indices = sorted(all_indexes - set(correct_indices))
      slice_f2 = fbeta_score(true_pred[correct_indices], test_pred[correct_indices], average='binary', beta=2)
      slice_precision = precision_score(true_pred[correct_indices], test_pred[correct_indices])
      slice_recall = recall_score(true_pred[correct_indices], test_pred[correct_indices])
      all_f2 = fbeta_score(true_pred[non_slice_indices], test_pred[non_slice_indices], average='binary', beta=2)
      all_precision = precision_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
      all_recall = recall_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
      if Metric == 'f2':
        f2_improvment = all_f2 - original_f2
      elif Metric == 'recall':
        f2_improvment = all_recall - original_recall
      res.append((data.columns[0],node_id+1, slice_f2, f2_improvment, slice_precision, slice_recall, size / max_index_original * 100, node_min, node_max, -1000,float('inf')))
    return res
  else: # two featuers
    ranges = sorted(create_ranges_2_features(tree), key=lambda x: x[0])
    for node_id in range(len(ranges)):
      node_min_f1 = ranges[node_id][1]
      node_max_f1 = ranges[node_id][2]
      node_min_f2 = ranges[node_id][3]
      node_max_f2 = ranges[node_id][4]
      correct_indices = data[(data[data.columns[0]].between(node_min_f1, node_max_f1)) &(data[data.columns[1]].between(node_min_f2, node_max_f2))].index
      size = len(orifinal_data[(orifinal_data[orifinal_data.columns[0]].between(node_min_f1, node_max_f1)) &(orifinal_data[orifinal_data.columns[1]].between(node_min_f2, node_max_f2))].index)
      non_slice_indices = sorted(all_indexes - set(correct_indices))
      slice_f2 = fbeta_score(true_pred[correct_indices], test_pred[correct_indices], average='binary', beta=2)
      slice_precision = precision_score(true_pred[correct_indices], test_pred[correct_indices])
      slice_recall = recall_score(true_pred[correct_indices], test_pred[correct_indices])
      all_f2 = fbeta_score(true_pred[non_slice_indices], test_pred[non_slice_indices], average='binary', beta=2)
      all_precision = precision_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
      all_recall = recall_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
      if Metric == 'f2':
        f2_improvment = all_f2 - original_f2
      elif Metric == 'recall':
        f2_improvment = all_recall - original_recall
      res.append((data.columns[0] +'-'+ data.columns[1], node_id+1, slice_f2, f2_improvment, slice_precision, slice_recall, size / max_index_original * 100, node_min_f1, node_max_f1, node_min_f2, node_max_f2))
    return res

  thresholds = [sorted(tree.threshold[tree.feature==0])]
  if number_of_features == 2:
      thresholds.append(sorted(tree.threshold[tree.feature==1]))
  leafs_indices = set(m.apply(data))
  leaf_list = m.apply(data)
  min__leaf = float('inf')
  min_leaf_perf = float('inf')
  for leaf_idx in leafs_indices:
    if tree.n_node_samples[leaf_idx] > min_sample_number and tree.impurity[leaf_idx] < max_impurity:
      indices = tree.value[leaf_idx]
      acc = indices[0][1] / sum(indices[0])
      true_prediction = indices[0][1]
      false_prediction = indices[0][0]
      if (True):
          leaf = leaf_list == leaf_idx
          samples_in_leaf = np.where(leaf)[0]
          data_in_leaf = data.values[samples_in_leaf]
          if data_in_leaf.size == 0:
              print("Leaf node is empty.")
          else:
              # Analyze the range of values for each feature
              minVal = [-1000 for x in range(number_of_features)]
              maxVal = [float('inf') for x in range(number_of_features)]
              for f in range(number_of_features):
                for th in range(len(thresholds[f])):
                  if thresholds[f][th]!=-2:
                    if data_in_leaf[0][f] < thresholds[f][th]:
                      #res.append((leaf_idx, acc,data.columns[0], float('-inf'), thresholdsA[th]))
                        maxVal[f] = thresholds[f][th]
                        #print("leaf idx " + str(leaf_idx) + " accuracy is " + str(acc) + " " + str(data.columns[0]) + " range: smaller than" +  str(thresholds[f][th]))
                        break
                    elif data_in_leaf[0][f] > thresholds[f][th]:
                      if th == len(thresholds[f])-1:
                        #res.append((leaf_idx, acc,data.columns[0], thresholdsA[th], float('inf')))
                        minVal[f] = thresholds[f][th]
                        #print("leaf idx " + str(leaf_idx) + " accuracy is " + str(acc) + " " + str(data.columns[0]) + " range: larger than" +  str(thresholds[f][th]))
                        break
                      elif data_in_leaf[0][f] < thresholds[f][th+1]:
                        #res.append((leaf_idx, acc,data.columns[0], thresholdsA[th], thresholdsA[th+1]))
                        minVal[f] = thresholds[f][th]
                        maxVal[f] = thresholds[f][th+1]
                        #print("leaf idx " + str(leaf_idx) + " accuracy is " + str(acc) + " " + str(data.columns[0]) + " range: " +  str(thresholds[f][th]) + " , " + str(thresholds[f][th+1]))
                        break
              feature_ranges = [(data_in_leaf[:, i].min(), data_in_leaf[:, i].max()) for i in range(data_in_leaf.shape[1])]

              orig_recall = recall_score(true_pred, test_pred)
              orig_precision = precision_score(true_pred, test_pred)
              original_f2 = fbeta_score(true_pred, test_pred, average='binary', beta=2)
              max_index  = data.index.max()
              all_indexes = set(range(max_index + 1))
              if number_of_features == 1:
                correct_indices = data[(data[data.columns[0]].between(feature_ranges[0][0], feature_ranges[0][1]))].index
                non_slice_indices = sorted(all_indexes - set(correct_indices))
                slice_f2 = fbeta_score(true_pred[correct_indices], test_pred[correct_indices], average='binary', beta=2)
                slice_precision = precision_score(true_pred[correct_indices], test_pred[correct_indices])
                slice_recall = recall_score(true_pred[correct_indices], test_pred[correct_indices])
                all_f2 = fbeta_score(true_pred[non_slice_indices], test_pred[non_slice_indices], average='binary',beta=2)
                all_precision = precision_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                all_recall = recall_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                f2_improvment = all_f2 - original_f2
                res.append((leaf_idx, slice_f2,f2_improvment,slice_precision, slice_recall, len(correct_indices),true_prediction,false_prediction,data.columns[0], minVal[0],maxVal[0]))
              else:
                correct_indices = data[(data[data.columns[0]].between(feature_ranges[0][0], feature_ranges[0][1])) & (data[data.columns[1]].between(feature_ranges[1][0], feature_ranges[1][1]))].index
                non_slice_indices = sorted(all_indexes - set(correct_indices))
                slice_f2 = fbeta_score(true_pred[correct_indices], test_pred[correct_indices], average='binary', beta=2)
                slice_precision = precision_score(true_pred[correct_indices], test_pred[correct_indices])
                slice_recall = recall_score(true_pred[correct_indices], test_pred[correct_indices])
                all_f2 = fbeta_score(true_pred[non_slice_indices], test_pred[non_slice_indices], average='binary',beta=2)
                all_precision = precision_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                all_recall = recall_score(true_pred[non_slice_indices], test_pred[non_slice_indices])
                f2_improvment = all_f2 - original_f2
                res.append((leaf_idx, slice_f2,f2_improvment,slice_precision, slice_recall, len(correct_indices),true_prediction,false_prediction,data.columns[0], minVal[0],maxVal[0],data.columns[1], minVal[1],maxVal[1]))
              if slice_f2 < min_leaf_perf:
                min_leaf = leaf_idx
                min_leaf_perf = slice_f2

  return res



def main_func_max(model_name, test_data, pred_test, y_test, number_of_features = 1, max_models_number = float('inf'), metric = 'TP'):
  res = []
  output_name = model_name
  for i in range(number_of_features):
    number_of_features = i + 1
    if metric == 'accuracy':
      relevant_predictions = (np.array(pred_test).flatten() ==  np.array(y_test))
      relevant_indices = [True for x in range(len(relevant_predictions))]
      test_x_ = test_data
      result, order = CreateModels(test_x_ , relevant_predictions, number_of_features, max_models_number)
      pred_test1 = pred_test
      y_test1 = y_test
    elif metric == 'f2':
      #keep only predictions for anomalies
      relevant_indices = ((np.array(pred_test).flatten() ==  np.array(y_test)) & (np.array(pred_test).flatten() == 1)) | (np.array(pred_test).flatten() !=  np.array(y_test))
      test_x_ = test_data[relevant_indices==True].reset_index(drop=True)
      relevant_predictions = (np.array(pred_test).flatten() ==  np.array(y_test))[relevant_indices==True]
      result, order = CreateModels(test_x_ , relevant_predictions, number_of_features, max_models_number)
      pred_test1 = pred_test[relevant_indices==True]
      y_test1 = y_test[relevant_indices==True]
      #relevant_predictions = np.array(pred_test).flatten()[np.array(y_test)!=0]
      #test_x_ = test_data[np.array(y_test)!=0]
      #result, order = CreateModels(test_x_ , relevant_predictions, number_of_features, max_models_number)

    worst_model_idx = None
    worst_model_imp = float('-inf')
    for f in range(min(max_models_number,len(order))):
      model = result[f]## model returned from the classifier
      original_acc =  sum(relevant_predictions) / len(relevant_predictions)
      original_f2 = fbeta_score(y_test1, pred_test1, average='binary', beta=2)
      acc_deg = 0.05
      min_sample_number = 0.1* len(y_test1)
      max_impurity = 1
      if number_of_features == 1:
        test_features = test_x_[[order[f][0]]]
        original_test_data = test_data[[order[f][0]]]
      else:
        test_features = test_x_[[order[f][0],order[f][1]]]
        original_test_data = test_data[[order[f][0],order[f][1]]]
      new_res = saveRelevantInfo2_no_acc(model,pred_test1,y_test1,original_test_data, test_features, original_acc,acc_deg,min_sample_number,max_impurity,number_of_features, 'f2')
      res.extend(new_res)


  #sort based on acc imp
  # = sorted(res, key=lambda x: x[4], reverse=True)

  df = pd.DataFrame(res, columns=['features', 'leaf id', 'f2','f2 imp', 'precision', 'recall', 'size','min val1', 'max val1', 'min val2', 'max val2'])
  filtered_size = df[df['size'] <= 20]
  filtered_df = filtered_size.drop_duplicates(subset=['min val1', 'max val1', 'min val2', 'max val2'])
  result_2 = filtered_df.sort_values(by=['features', 'f2 imp'], ascending=False)
  result1 = result_2.loc[result_2.groupby('features')['f2 imp'].idxmax()]
  df_sortedFull = result1.sort_values(by='f2 imp', ascending=False)
  print("\nUsing IPython.display.display():")
  df_sorted = df_sortedFull.head(25)
  display(df_sorted)


  # Plotting the bar chart
  plt.figure(figsize=(10, 6))
  bar_width = 0.1  # Change the width as needed
  bars = plt.bar(df_sorted['size'], df_sorted['f2 imp'], width=bar_width,color='skyblue', alpha=0.5)  # Use 'size' instead of 'features'
  #bars = plt.bar(df_sorted['size'], df_sorted['acc imp'])

  y_coordinate = min(df_sorted['f2 imp'])  # Start from the bottom of the first bar


  # Add feature name inside each bar, vertically
  for bar, feature in zip(bars, df_sorted['features']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), feature,
            ha='center', va='center', rotation=90)

  plt.xlabel('slice percentage size')
  plt.ylabel('Potential f2 Improvement')
  plt.title('Potential f2 improvement for identified slice')
  plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
  plt.tight_layout()  # Adjust layout to prevent clipping of labels
  plt.show()
  plt.savefig(output_name +  'sizePlot.jpg')

  # Plotting the bar chart
  plt.figure(figsize=(10, 6))
  plt.bar(df_sorted['features'], df_sorted['f2 imp'])
  plt.xlabel('Feature')
  plt.ylabel('Potential f2 Improvement')
  plt.title('Potential f2 improvement for identified slice')
  plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
  plt.tight_layout()  # Adjust layout to prevent clipping of labels
  plt.show()
  plt.savefig(output_name +  'FeaturesPlot.jpg')

  worst_feature = df_sorted.iloc[0]['features']
  worst_feature_df = filtered_df[filtered_df['features'] == worst_feature]
  worst_feature_df1 = worst_feature_df.sort_values(by='f2 imp', ascending=False)
  # Plotting the bar chart
  plt.figure(figsize=(10, 6))
  plt.bar(worst_feature_df1['leaf id'], worst_feature_df1['f2 imp'])
  plt.xlabel('leaf id')
  plt.ylabel('Potential f2 Improvement')
  plt.title('Potential f2 improvement for identified slices -' + worst_feature)
  plt.xticks(worst_feature_df1['leaf id'], rotation=90)
  plt.tight_layout()  # Adjust layout to prevent clipping of labels
  plt.show()
  plt.savefig(output_name +  'WorstSlicePlot.jpg')
  plt.savefig('plot.png')

  df_sorted.to_csv(output_name + 'output.csv', index=False)

  return df_sorted

def train_percentage(test_x, df):
  for i in range(2):
    first_row_values = df.iloc[i]
    feature = first_row_values['feature']
    min_val1 = first_row_values['min val1']
    max_val1 = first_row_values['max val1']
    feature_2 = first_row_values['feature_2']
    min_val2 = first_row_values['min val2']
    max_val2 = first_row_values['max val2']
    filtered_data = test_x[(test_x[feature] >= min_val1) & (test_x[feature] <= max_val1)]
    # Calculate the number of rows within the range
    count_within_range = len(filtered_data)
    # Calculate the total number of rows in the DataFrame
    total_count = len(test_x)
    # Calculate the percentage of rows within the range
    percentage = (count_within_range / total_count) * 100
    if i == 0:
      print("max slide percentage")
    else:
      print("min slide percentage")
    print(percentage)
