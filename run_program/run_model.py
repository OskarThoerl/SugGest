"""
Live Gesture Recognition with Myo Armband by Oskar Thörl and Nadja Bobić

This script handles all elements and interactions around the data gathering and gesture recognition using a Myo armband.
It uses the Myo SDK, numpy, and keras libraries to gather and preprocess the data, load the trained LSTM model, and make predictions.

The script contains several functions:

The on_keypress function initializes the global key variable and starts the keyboard listener.
The EmgCollector class handles all the data gathering. It initializes all variables,
    measures Myo EMG data and adds it to the queue, prints confirmation message and starts data stream when connected,
    prints disconnection message and sets myo_status to False on disconnect,
    adds EMG Data to the emg_data list when new data is received, adds acceleration, gyroscope and orientation Data
    to the corresponding list when new data is received, and runs the main function which initializes the user interface,
    starts the datastream, creates a new frame and adds it to the batch, matches the current prediction to the possible cases
    and displays it if a threshold is reached.
The main function starts the EMGCollector.

This EMGCollector class uses the UI class and its functions.
"""

# Initialize Libraries
from time import time
from threading import Lock
from keras import models
import numpy as np
import myo
from user_interface import UI
import keyboard
import sys

# Initialize the global key variable and start the keyboard listener
key = "NaN"
def on_keypress(event):
    global key
    key = event.name
keyboard.on_press(on_keypress)

# Class that handles all the data gathering
class EmgCollector(myo.DeviceListener):

    # Initialize all variables
    def __init__(self, n, path):
        self.n = n
        self.lock = Lock()
        self.start_time = 0
        self.batch = []
        self.model = models.load_model(path)
        self.emg_data = []
        self.acceleration_data = []
        self.gyroscope_data = []
        self.orientation_data = []
        self.prev_predictions = []

    # Measure Myo EMG data and ad it to the queue
    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # Print confirmation message and start data stream when connected
    def on_connected(self, event):
        print("Myo connected...")
        self.myo_status = True
        event.device.stream_emg(True)

    # print disconnection message and set myo_status to False on disconnect
    def on_disconnected(self, event):
        print("Myo disconnected...")
        self.myo_status = False

    # Add EMG Data to the emg_data list when new data is received
    def on_emg(self, event):
        with self.lock:
            self.emg_data = list(event.emg)

    # Add acceleration, gyroscope and orientation Data to the corresponding list when new data is received
    def on_orientation(self, event):
        with self.lock:
            self.acceleration_data = list(event.acceleration)
            self.gyroscope_data = list(event.gyroscope)
            self.orientation_data = list(event.orientation)

    def main(self):
        print("running main")
        # Initialize the myo sdk, start up the hub
        myo.init(sdk_path=r'C:\Program Files (x86)\myo-sdk-win-0.9.0')
        hub = myo.Hub()

        # Initialize helper variables
        counter = 0
        start_time = time()

        # Initialize user interface
        ui = UI()

        # Start the datastream
        with hub.run_in_background(self.on_event):
            while True:
                # Run approximately every 5ms
                if (time()-start_time) >= 0.0049:
                    # Update the helper variables
                    start_time = time()
                    counter += 1
                    # Create a new frame (one_line) and add it to the batch
                    one_line = np.array(self.emg_data + self.acceleration_data + self.gyroscope_data + self.orientation_data)
                    if len(one_line) == 18:
                        self.batch.append(one_line)

                    # Run every 10th cycle (every 50ms)
                    if counter%10 == 0:
                        # End the program if the Escape key is pressed
                        if key == 'esc':
                            sys.exit()
                        # Ensure the batch has size 100
                        while len(self.batch) > 100:
                            self.batch.remove(self.batch[0])
                        if len(self.batch) > 99:
                            # Reshape batch and get a prediction
                            batch_list = list(self.batch)
                            np_batch = np.array(batch_list)
                            np_batch = np.reshape(np_batch, (1, np_batch.shape[0], np_batch.shape[1]))
                            prediction = int(self.model.predict(np_batch, verbose=0).argmax())
                            # Append prediction to the previous predictions and shorten them to a length of 10
                            self.prev_predictions.append(prediction)
                            while len(self.prev_predictions) > 10:
                                del self.prev_predictions[0]
                            # Match the current prediction to the possible cases and display it if a threshold is reached
                            match prediction:
                                case 0:
                                    print("consume")
                                    if self.prev_predictions.count(prediction) >= 5:    # Check if threshold is reached
                                        self.prev_predictions = []
                                        self.batch = []
                                        ui.display("consume")
                                case 1:
                                    print("sleep")
                                    if self.prev_predictions.count(prediction) >= 5:    # Check if threshold is reached
                                        self.prev_predictions = []
                                        self.batch = []
                                        ui.display("sleep")
                                case 2:
                                    print("attention")
                                    if self.prev_predictions.count(prediction) >= 8:    # Check if threshold is reached
                                        self.prev_predictions = []
                                        self.batch = []
                                        ui.display("attention")
                                case 3:
                                    print("moving")
                                    pass
                                case 4:
                                    print("inactive")
                                    pass



def main():
    # Start the EMGCollector
    print("running system main")
    listener = EmgCollector(1, '100_set_model1.keras')
    listener.main()


if __name__ == '__main__':
    main()