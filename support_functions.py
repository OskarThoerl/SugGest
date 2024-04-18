# Initialize Libraries
import os
import pandas as pd
import numpy as np
import random

# Initialize directory of all the files used
data_dir = 'data_recordings'
# Directories dictionary -- allows for easy inclusion of additional file directories
directories = {
    'dir': data_dir,
}

# Initializing global variables
X_full = []
y_full = []
count = {
    'a':0,
    'm':0,
    'i':0,
    'c':0,
    's':0
}

# Function to preprocess one csv file
def preprocess(file_path):
    # Accessing the global variables
    global X_full
    global y_full

    # Loading the csv file
    df = pd.read_csv(file_path)

    # Assign raw values to 'X', 'y' and 'timestamps'
    X = df.drop(['timestamp', 'label'], axis=1)
    X = X.to_numpy()
    y_char = df['label']
    timestamps = df['timestamp'].to_numpy()

    # Initialize helper variables
    length_counter = 0
    prev_time = 0
    one_batch = []

    # Initialize updated X and y
    new_X = []
    new_y = []

    # Pass through entire X and y
    for i in range(len(timestamps)):
        if prev_time > timestamps[i]:       # Detect if a new gesture starts
            while length_counter < 400:     # Ensure that all batches are of length 400
                # Append data to 'one_batch'
                one_batch.append(X[i-1])
                length_counter += 1
            # Append one batch to the updated X variable and the corresponding label to the updated y
            new_y.append(y_char[i - 1])
            new_X.append(one_batch)
            # Reset helper variables
            one_batch = []
            length_counter = 0
        if length_counter < 400:            # Ensure that all batches are of length 400
            one_batch.append(X[i])
            length_counter += 1
        # Update the previous time to be able to compare times
        prev_time = timestamps[i]
    # Ensure that the last batch is not disregarded
    if one_batch != []:
        while len(one_batch)%400 != 0:
            one_batch.append(one_batch[-1])
        new_X.append(one_batch)
        new_y.append(new_y[-1])

    # Convert labels from letters to numbers (d = drink, e = eat and c = consume were combined)
    y = []
    for char in new_y:
        match char:
            case 'd':
                y.append(0)
                count['c'] += 1
            case 'e':
                y.append(0)
                count['c'] += 1
            case 'c':
                y.append(0)
                count['c'] += 1
            case 's':
                y.append(1)
                count['s'] += 1
            case 'a':
                y.append(2)
                count['a'] += 1
            case 'm':
                y.append(3)
                count['m'] += 1
            case 'i':
                y.append(4)
                count['i'] += 1

    # Turn X and y into numpy arrays
    X = np.array(new_X)
    y = np.array(y)

    # Append X and y to the global X and y variables to combine all csv files in one dataset
    X_full.extend(X)
    y_full.extend(y)

# Function to call preprocessing function for all files in the given directories
def preprocess_all_data():
    # Accessing the global variables
    global X_full
    global y_full
    # Iterating through all files in the given directory and getting their paths
    for idx, (n, d) in enumerate(directories.items()):
        filepath = d + "/"
        filelist = [(filepath + f) for f in os.listdir(filepath)]
        for file in filelist:
            # Call the preprocess function for all files
            preprocess(file)
    # Ensure X_full and y_full are numpy arrays, print their shape and return them
    X_full = np.array(X_full)
    y_full = np.array(y_full)
    print(np.shape(X_full), np.shape(y_full))
    print(count)
    return X_full, y_full

# Function to split data into train test and validation groups
def train_test_val_split(X, y, test_perc = 0.2, val_perc = 0.2):
    # Get the total amount of sets
    tot_sets = len(X)

    # Calculate amount of test and validation sets
    X_test = []
    y_test = []
    amount_test_sets = int(tot_sets * test_perc)
    X_val = []
    y_val = []
    amount_val_sets = int(tot_sets * val_perc)
    # randomly assign sets to test and validation groups
    val_test_sets = random.sample(range(tot_sets), amount_test_sets+amount_val_sets)

    # Assign part of the sets according to the random numbers to the test group
    for element in val_test_sets[:amount_test_sets]:
        X_test.append(X[element])
        y_test.append(y[element])

    # Assign other part of sets according to the random numbers to the validation group
    for element in val_test_sets[amount_test_sets:]:
        X_val.append(X[element])
        y_val.append(y[element])

    # Sort sets in descending order to avoid mistakes when deleting elements
    val_test_sets.sort(reverse=True)
    # Assign all sets to X_train and y_train and delete all sets used by the test and validation group
    X_train = list(X)
    y_train = list(y)
    for element in val_test_sets:
        del X_train[element]
        del y_train[element]

    # Convert all variables to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    # Print shapes and return groups
    print(X.shape, X_train.shape, X_test.shape, X_val.shape, y.shape, y_train.shape, y_test.shape, y_val.shape)
    return X_train, y_train, X_test, y_test, X_val, y_val

# Function to split data sets into smaller sets of a given length
def LSTR_sequence_split(X, y, look_back = 50):
    # Initialize new X and y
    new_X = []
    new_y = []

    # Loop through all sets
    for i in range(len(X)):
        one_gesture = list(X[i])
        while len(one_gesture) >= look_back:
            new_X.append(one_gesture[:look_back])
            new_y.append(y[i])
            try:
                del one_gesture[0]
            except:
                pass
    new_X = np.array(new_X)
    new_y = np.array(new_y)
    print(new_X.shape, new_y.shape)
    return new_X, new_y

def remove_overflow(X, y, batch_size):
  num_samples = X.shape[0]
  overflow = num_samples % batch_size
  if overflow > 0:
    shortened_X = X[:-overflow]
    shortened_y = y[:-overflow]
  else:
    shortened_X = X
    shortened_y = y
  print(shortened_X.shape, shortened_y.shape)
  return shortened_X, shortened_y