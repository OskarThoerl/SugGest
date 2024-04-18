"""
LSTM Model Training by Oskar Thörl and Nadja Bobić

This is the main script for training an LSTM model for gesture recognition using time-series data from a Myo armband.
It handles all elements and interactions around the model training process.

It preprocesses the data, defines the model architecture (including normalization, LSTM, and dense layers),
compiles the model, and trains it over a specified number of iterations.
The training history is plotted and the final model is saved.

This script contains several functions:
- The 'plot_confusion' function for generating and displaying a confusion matrix.
- The 'train_LSTM' function for training the LSTM model from scratch.
- The 'continue_training' function for continuing training on an existing model.
- The 'save_model' and 'load_model' functions for saving a trained model to disk and loading it back into memory, respectively.

This script calls functions from the libraries that are initialized hereafter
"""

# Initialize Libraries
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, utils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from support_functions import preprocess_all_data, train_test_val_split, LSTR_sequence_split, remove_overflow


# Function to plot a confusion matrix
def plot_confusion(X_test, y_test, model, title):
    # Get predicted and actual labels in correct format
    y_predict = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Generate and display confusion matrix
    result = confusion_matrix(y_true, y_predict, labels=[0,1,2,3,4])
    disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=['c','s','a','m','i'])
    disp.plot()
    plt.title(title)

# Function for training an LSTM
def train_LSTM():
    # Define main variables
    look_back = 100     # Length of sequences
    batch_size = 100    # Batch size (how many sequences the Ai trains on in parallel)
    iters = 50          # Iterations --> Number of training cycles

    # Get data
    X, y = preprocess_all_data()

    # Split data into training, test and validation groups
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(X, y)

    # Transform all sets of all groups into multiple sets of look_back size
    X_train, y_train = LSTR_sequence_split(X_train, y_train, look_back)
    X_test, y_test = LSTR_sequence_split(X_test, y_test, look_back)
    X_val, y_val = LSTR_sequence_split(X_val, y_val, look_back)

    # Reshape all groups to fit the LSTM input shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 18))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 18))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 18))

    # Turn labels into categorical labels (0 --> [1, 0, 0, 0, 0]; 1 --> [0, 1, 0, 0, 0]; etc.)
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    y_val = utils.to_categorical(y_val)

    # Define / Build the Neural Network
    model = models.Sequential()
    # Layer for input and normalization of data
    model.add(layers.Normalization(axis=-1, input_shape=(look_back, 18), mean=None, variance=None, invert=False))
    # Two LSTM layers to process time-series data
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.LSTM(50, return_sequences=False))
    # Two dense layers for complex pattern detection
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    # Dense output layer for outputting an array of 5 --> probability for each label
    model.add(layers.Dense(5, activation='softmax'))

    # Compile the model using an optimizer and a loss function
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Train the model
    hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}     # History for recording progress
    for i in range(iters):
        print("{} of {}".format(i+1, iters))
        epoch_hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                               epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        for key in hist.keys():
            hist[key].append(epoch_hist.history[key])

        for layer in model.layers:
            if layer.name == 'lstm':
                layer.reset_states()

    # Plot accuracy vs validation accuracy over time (taken from programming tutorial)
    plt.plot(hist['accuracy'], label='accuracy')
    plt.plot(hist['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Print final loss and accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
    print("Loss: {}; Accuracy: {}".format(test_loss, test_acc))

    # Plot the confusion matrix
    plot_confusion(X_val, y_val, model, "LSTM")

    # Save and return the model
    save_model(model)
    return model

# Function for continuing training on an existing model (very similar to "train_LSTM()")
def continue_training(model_name):
    # Define main variables
    look_back = 100  # Length of sequences
    batch_size = 100  # Batch size (how many sequences the Ai trains on in parallel)
    iters = 50  # Iterations --> Number of training cycles

    # Get data
    X, y = preprocess_all_data()

    # Split data into training, test and validation groups
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(X, y)

    # Transform all sets of all groups into multiple sets of look_back size
    X_train, y_train = LSTR_sequence_split(X_train, y_train, look_back)
    X_test, y_test = LSTR_sequence_split(X_test, y_test, look_back)
    X_val, y_val = LSTR_sequence_split(X_val, y_val, look_back)

    # Reshape all groups to fit the LSTM input shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 18))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 18))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 18))

    # Turn labels into categorical labels (0 --> [1, 0, 0, 0, 0]; 1 --> [0, 1, 0, 0, 0]; etc.)
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    y_val = utils.to_categorical(y_val)

    # load the existing model
    model = models.load_model(model_name)

    # Train the model
    hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}  # History for recording progress
    for i in range(iters):
        print("{} of {}".format(i + 1, iters))
        epoch_hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                               epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        for key in hist.keys():
            hist[key].append(epoch_hist.history[key])

        for layer in model.layers:
            if layer.name == 'lstm':
                layer.reset_states()

    # Plot accuracy vs validation accuracy over time (taken from programming tutorial)
    plt.plot(hist['accuracy'], label='accuracy')
    plt.plot(hist['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Print final loss and accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
    print("Loss: {}; Accuracy: {}".format(test_loss, test_acc))

    # Plot the confusion matrix
    plot_confusion(X_val, y_val, model, "LSTM")

    # Save and return the model
    save_model(model)
    return model

# Function for saving the model
def save_model(model):
    model.save("model1.keras")

# Function for loading an existing model
def load_model():
    model = models.load_model("model1.keras")
    return model
