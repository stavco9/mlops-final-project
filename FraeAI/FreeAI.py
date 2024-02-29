
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.tree import DecisionTreeClassifier
import itertools
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import plot_tree
from IPython.display import display


def CreateModels(test_data , accurate_prediction, number_of_features, print_tree = False, max_models_number = float('inf')):
  #list of all the returned modules
  models_res = []
  categories_list = list(test_data)
  num_of_models = 0
  #saves the feature/s of the trees which are being created
  models_order =  list(itertools.combinations(categories_list, number_of_features))
  #create all the trees
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

    #create Decision tree
    model = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=1)
    model.fit(X, accurate_prediction)

    #only for debugging
    if print_tree == True:
      plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
      plot_tree(model, filled=True, feature_names=f_names, node_ids=True)
      plt.savefig(f'{f_names}' + '.png')
      plt.clf()

    #add tree to returned trees
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
def FilterTreeNodes(model,test_pred1, true_pred, orifinal_data, data, min_sample_number, max_impurity, number_of_features = 1, Metric = 'f2'):
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
    node_data_ranges = sorted(create_ranges(tree), key=lambda x: x[0])
    #for each node run calculate the potential f2 improvement
    for node_id in range(len(node_data_ranges)):
      node_min = node_data_ranges[node_id][1]
      node_max = node_data_ranges[node_id][2]
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
    node_data_ranges = sorted(create_ranges_2_features(tree), key=lambda x: x[0])
    #for each node run calculate the potential f2 improvement
    for node_id in range(len(node_data_ranges)):
      node_min_f1 = node_data_ranges[node_id][1]
      node_max_f1 = node_data_ranges[node_id][2]
      node_min_f2 = node_data_ranges[node_id][3]
      node_max_f2 = node_data_ranges[node_id][4]
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

def main_func(model_name, test_data, pred_test, y_test, number_of_features = 1, max_models_number = float('inf'), metric = 'TP'):
  res = []
  output_name = model_name
  #create trees for all single and couple of features
  for features_num in range(1,number_of_features+1):
    if metric == 'accuracy':
      relevant_predictions = (np.array(pred_test).flatten() ==  np.array(y_test))
      test_x = test_data
      pred_test1 = pred_test
      y_test1 = y_test
      #create the models
      result, order = CreateModels(test_x , relevant_predictions, features_num, False, max_models_number)
    elif metric == 'f2':
      #remove true negative when building the tree
      relevant_indices = ((np.array(pred_test).flatten() ==  np.array(y_test)) & (np.array(pred_test).flatten() == 1)) | (np.array(pred_test).flatten() !=  np.array(y_test))
      test_x = test_data[relevant_indices==True].reset_index(drop=True)
      relevant_predictions = (np.array(pred_test).flatten() ==  np.array(y_test))[relevant_indices==True]
      #create the models
      result, order = CreateModels(test_x , relevant_predictions, features_num,False,max_models_number)
      pred_test1 = pred_test[relevant_indices==True]
      y_test1 = y_test[relevant_indices==True]

    worst_model_idx = None
    worst_model_imp = float('-inf')
    #iterate over all returned trees and keep the relevant nodes
    for f in range(min(max_models_number,len(order))):
      model = result[f]## model returned from the classifier

      min_sample_number = 0.1* len(y_test1)
      max_impurity = 1
      if features_num == 1:
        test_features = test_x[[order[f][0]]]
        original_test_data = test_data[[order[f][0]]]
      else:
        test_features = test_x[[order[f][0],order[f][1]]]
        original_test_data = test_data[[order[f][0],order[f][1]]]
      new_res = FilterTreeNodes(model,pred_test1,y_test1,original_test_data, test_features,min_sample_number,max_impurity,features_num, 'f2')
      res.extend(new_res)

  #create data frame from the saved nodes.
  df = pd.DataFrame(res, columns=['features', 'node id', 'f2','f2 imp', 'precision', 'recall', 'size','min val1', 'max val1', 'min val2', 'max val2'])
  #filter large data slices
  filtered_size = df[df['size'] <= 20]
  filtered_df = filtered_size.drop_duplicates(subset=['min val1', 'max val1', 'min val2', 'max val2'])
  result_2 = filtered_df.sort_values(by=['features', 'f2 imp'], ascending=False)
  result1 = result_2.loc[result_2.groupby('features')['f2 imp'].idxmax()]
  df_sortedFull = result1.sort_values(by='f2 imp', ascending=False)
  print("\nUsing IPython.display.display():")
  df_sorted = df_sortedFull.head(25)
  display(df_sorted)


  #bar plot for potential f2 improvement and data slices.
  #for similar potential improvement the customer may be intersted in smaller data slices
  plt.figure(figsize=(10, 6))
  bar_width = 0.1  # Change the width as needed
  bars = plt.bar(df_sorted['size'], df_sorted['f2 imp'], width=bar_width,color='skyblue', alpha=0.5)
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

  #bar plot for the slides with the maximal f2 potential improvement per feature/couple of features
  plt.figure(figsize=(10, 6))
  plt.bar(df_sorted['features'], df_sorted['f2 imp'])
  plt.xlabel('Feature')
  plt.ylabel('Potential f2 Improvement')
  plt.title('Potential f2 improvement for identified slice')
  plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
  plt.tight_layout()  # Adjust layout to prevent clipping of labels
  plt.show()
  plt.savefig(output_name +  'FeaturesPlot.jpg')

  #bar plot with the potential f2 improvement per data slice for the feature with the maximal potential f2 improvement
  worst_feature = df_sorted.iloc[0]['features']
  worst_feature_df = filtered_df[filtered_df['features'] == worst_feature]
  worst_feature_df1 = worst_feature_df.sort_values(by='f2 imp', ascending=False)
  plt.figure(figsize=(10, 6))
  plt.bar(worst_feature_df1['node id'], worst_feature_df1['f2 imp'])
  plt.xlabel('node id')
  plt.ylabel('Potential f2 Improvement')
  plt.title('Potential f2 improvement for identified slices -' + worst_feature)
  plt.xticks(worst_feature_df1['node id'], rotation=90)
  plt.tight_layout()  # Adjust layout to prevent clipping of labels
  plt.show()
  plt.savefig(output_name +  'WorstSlicePlot.jpg')
  plt.savefig('plot.png')

  #saves the output report to a csv
  df_sorted.to_csv(output_name + 'output.csv', index=False)

  return result1

