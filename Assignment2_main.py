import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import Create_Dataset
from k_means import KMeans
from kNN import kNN

def accuracy(outputs_TS, predictions):
    return 100*np.sum(outputs_TS == predictions) / len(outputs_TS)

# Define normalization function
def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

#Create_Dataset.create_dataset()

#Decide operating state
'''
    operating state:
    high load = 1
    low load = 2
    generator disconnection = 3
    line disconnection = 4
'''

# Generate training data
# High load
operating_state = 1
vm_pu_tot = pd.read_excel('time_series_training/high_load/res_bus/vm_pu.xlsx')
va_degree_tot = pd.read_excel('time_series_training/high_load/res_bus/va_degree.xlsx')

# Get the voltage magnitude and angle from the data frame
vm_pu_tot.drop(vm_pu_tot.columns[0], axis = 1, inplace = True)
va_degree_tot.drop(va_degree_tot.columns[0], axis = 1, inplace = True)

voltage_training = pd.concat([vm_pu_tot, va_degree_tot], axis = 1)
voltage_training.apply(normalize_column)

# Use the operating state as the label
training_label1 = pd.DataFrame(np.full((len(voltage_training), 1),operating_state), columns = ['state'])
voltage_training1 = pd.concat([voltage_training, training_label1], axis = 1)


#Low load
operating_state = 2
vm_pu_tot = pd.read_excel('time_series_training/low_load/res_bus/vm_pu.xlsx')
va_degree_tot = pd.read_excel('time_series_training/low_load/res_bus/va_degree.xlsx')
vm_pu_tot.drop(vm_pu_tot.columns[0], axis = 1, inplace = True)
va_degree_tot.drop(va_degree_tot.columns[0], axis = 1, inplace = True)
voltage_training = pd.concat([vm_pu_tot, va_degree_tot], axis = 1)
voltage_training.apply(normalize_column)
training_label2 = pd.DataFrame(np.full((len(voltage_training), 1),operating_state), columns = ['state'])
voltage_training2 = pd.concat([voltage_training, training_label2], axis = 1)


#Generator disconnection
operating_state = 3
vm_pu_tot = pd.read_excel('time_series_training/generator_disconnection/res_bus/vm_pu.xlsx')
va_degree_tot = pd.read_excel('time_series_training/generator_disconnection/res_bus/va_degree.xlsx')
vm_pu_tot.drop(vm_pu_tot.columns[0], axis = 1, inplace = True)
va_degree_tot.drop(va_degree_tot.columns[0], axis = 1, inplace = True)
voltage_training = pd.concat([vm_pu_tot, va_degree_tot], axis = 1)
voltage_training.apply(normalize_column)
training_label3 = pd.DataFrame(np.full((len(voltage_training), 1),operating_state), columns = ['state'])
voltage_training3 = pd.concat([voltage_training, training_label3], axis = 1)


#Line disconnection
operating_state = 4
vm_pu_tot = pd.read_excel('time_series_training/line_disconnection/res_bus/vm_pu.xlsx')
va_degree_tot = pd.read_excel('time_series_training/line_disconnection/res_bus/va_degree.xlsx')
vm_pu_tot.drop(vm_pu_tot.columns[0], axis = 1, inplace = True)
va_degree_tot.drop(va_degree_tot.columns[0], axis = 1, inplace = True)
voltage_training = pd.concat([vm_pu_tot, va_degree_tot], axis = 1)
training_label4 = pd.DataFrame(np.full((len(voltage_training), 1),operating_state), columns = ['state'])
voltage_training4 = pd.concat([voltage_training, training_label4], axis = 1)

vol_training = pd.concat([voltage_training1, voltage_training2, voltage_training3, voltage_training4], axis = 0, ignore_index=True)
vol_training = vol_training.fillna(0)

vol_training_label = pd.concat([training_label1, training_label2, training_label3, training_label4], axis = 0, ignore_index=True)


# Generate test data
#High load
operating_state = 1
vm_pu_tot_test = pd.read_excel('time_series_test/high_load/res_bus/vm_pu.xlsx')
va_degree_tot_test = pd.read_excel('time_series_test/high_load/res_bus/va_degree.xlsx')
vm_pu_tot_test.drop(vm_pu_tot_test.columns[0], axis = 1, inplace = True)
va_degree_tot_test.drop(va_degree_tot_test.columns[0], axis = 1, inplace = True)

#Save the test sets with labels and without labels
voltage_test1 = pd.concat([vm_pu_tot_test, va_degree_tot_test], axis = 1)
voltage_test1.apply(normalize_column)
test_label1 = pd.DataFrame(np.full((len(voltage_test1), 1),operating_state), columns = ['state'])
voltage_test1_true = pd.concat([voltage_test1, test_label1], axis = 1)


#Low load
operating_state = 2
vm_pu_tot_test = pd.read_excel('time_series_test/low_load/res_bus/vm_pu.xlsx')
va_degree_tot_test = pd.read_excel('time_series_test/low_load/res_bus/va_degree.xlsx')
vm_pu_tot_test.drop(vm_pu_tot_test.columns[0], axis = 1, inplace = True)
va_degree_tot_test.drop(va_degree_tot_test.columns[0], axis = 1, inplace = True)
voltage_test2 = pd.concat([vm_pu_tot_test, va_degree_tot_test], axis = 1)
voltage_test2.apply(normalize_column)
test_label2 = pd.DataFrame(np.full((len(voltage_test2), 1),operating_state), columns = ['state'])
voltage_test2_true = pd.concat([voltage_test2, test_label2], axis = 1)


#Generator disconnection
operating_state = 3
vm_pu_tot_test = pd.read_excel('time_series_test/generator_disconnection/res_bus/vm_pu.xlsx')
va_degree_tot_test = pd.read_excel('time_series_test/generator_disconnection/res_bus/va_degree.xlsx')
vm_pu_tot_test.drop(vm_pu_tot_test.columns[0], axis = 1, inplace = True)
va_degree_tot_test.drop(va_degree_tot_test.columns[0], axis = 1, inplace = True)
voltage_test3 = pd.concat([vm_pu_tot_test, va_degree_tot_test], axis = 1)
voltage_test3.apply(normalize_column)
test_label3 = pd.DataFrame(np.full((len(voltage_test3), 1),operating_state), columns = ['state'])
voltage_test3_true = pd.concat([voltage_test3, test_label3], axis = 1)
    
    
#Line disconnection
operating_state = 4
vm_pu_tot_test = pd.read_excel('time_series_test/line_disconnection/res_bus/vm_pu.xlsx')
va_degree_tot_test = pd.read_excel('time_series_test/line_disconnection/res_bus/va_degree.xlsx')
vm_pu_tot_test.drop(vm_pu_tot_test.columns[0], axis = 1, inplace = True)
va_degree_tot_test.drop(va_degree_tot_test.columns[0], axis = 1, inplace = True)
voltage_test4 = pd.concat([vm_pu_tot_test, va_degree_tot_test], axis = 1)
voltage_test4.apply(normalize_column)
test_label4 = pd.DataFrame(np.full((len(voltage_test4), 1),operating_state), columns = ['state'])
voltage_test4_true = pd.concat([voltage_test4, test_label4], axis = 1)
    
    
vol_test = pd.concat([voltage_test1, voltage_test2, voltage_test3, voltage_test4], axis = 0, ignore_index=True)
vol_test = vol_test.fillna(0)
vol_test_label = pd.concat([test_label1, test_label2, test_label3, test_label4], axis = 0, ignore_index=True)
vol_test_labeled = pd.concat([voltage_test1_true, voltage_test2_true, voltage_test3_true, voltage_test4_true], axis = 0, ignore_index=True)
vol_test_labeled = vol_test_labeled.fillna(0)

'''
# Plot the voltage angle and magnitude
# Extract the values ​​of the first 9 columns and columns 10-18
y = vol_training.iloc[:, :9].values
x = vol_training.iloc[:, 9:18].values

# Get the value of the status column, used to determine the color
statuses = vol_training.iloc[:, -1]


# Create a colormap dictionary for selecting the corresponding color based on the statuses
color_map = {1: 'blue', 2: 'orange', 3: 'red', 4: 'green'}

for i in range(len(y)):
    
    current_y = y[i]
    current_x = x[i]
    
    status = statuses[i]
    color = color_map.get(status, 'gray')
    
    plt.scatter(current_x, current_y, color=color, s=5)

legend_labels = [
    mpatches.Patch(color='blue', label='High Load'),
    mpatches.Patch(color='orange', label='Low Load'),
    mpatches.Patch(color='red', label='Generator Disconnection'),
    mpatches.Patch(color='green', label='Line Disconnection')
]

plt.title('Training set Voltage')
plt.xlabel('Voltage Angele (degree)')
plt.ylabel('Voltage Magnitude (p.u.)')

plt.legend(handles=legend_labels)

plt.show()


vol_training.to_excel('training_set.xlsx')


# Generate data for K-Means model
train_features = vol_training.values[:, :-1]
train_labels = vol_training.values[:, -1]
test_features = vol_test.iloc[:, :-1].values

# Clustering using the K-means algorithm
clusters = len(np.unique(train_labels))
print("Total of clusters defined in dataset: ", clusters)
k = KMeans(max_iters=100)
y_pred, ks = k.predict(train_features)
print("Number of clusters based on k-means:", ks)
print("Contigency Table or Cross Table of 'True' Outputs and Prediction: ")
print(pd.crosstab(train_labels, y_pred))

'''
# K-NN algorithm
training_set = vol_training.drop(vol_training.columns[-1], axis=1).to_numpy()
training_label = vol_training.iloc[:, -1].to_numpy()
test_set = vol_test_labeled.drop(vol_test_labeled.columns[-1], axis=1).to_numpy()
test_label = vol_test_labeled.iloc[:, -1].to_numpy()

# Classification by K-NN
clf = kNN(k=5)
clf.fit(training_set, training_label)
predictions = clf.predict(test_set)
print("Prediction: ", predictions)
print("Test data: ", test_label)
print("Accuracy of classification: ", accuracy(test_label, predictions), "%")  


