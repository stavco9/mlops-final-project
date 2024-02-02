from General_modules.dataset import Dataset

dataset = Dataset('./SKAB')
dataset.display_data()
#dataset.show_plt_data()
dataset.show_heatmap_data()
dataset.split_data()
dataset.display_X()
#dataset.show_plt_free_anomaly()
dataset.standard_data()
#dataset.show_smooth_data()