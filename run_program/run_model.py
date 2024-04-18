from time import time
from threading import Lock
from keras import models
import numpy as np
import myo
from user_interface import UI
import keyboard
import sys

key = "NaN"


def on_keypress(event):
    global key
    key = event.name


keyboard.on_press(on_keypress)

class EmgCollector(myo.DeviceListener):

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

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def on_connected(self, event):
        print("Myo connected...")
        self.myo_status = True
        event.device.stream_emg(True)

    def on_disconnected(self, event):
        print("Myo disconnected...")
        self.myo_status = False

    def on_emg(self, event):
        with self.lock:
            self.emg_data = list(event.emg)

    def on_orientation(self, event):
        with self.lock:
            self.acceleration_data = list(event.acceleration)
            self.gyroscope_data = list(event.gyroscope)
            self.orientation_data = list(event.orientation)

    def main(self):
        print("running main")
        myo.init(sdk_path=r'C:\Program Files (x86)\myo-sdk-win-0.9.0')
        hub = myo.Hub()
        counter = 0
        last_run = 0
        start_time = time()
        ui = UI()
        with hub.run_in_background(self.on_event):
            while True:
                if (time()-start_time) >= 0.0049:
                    # print((time()-start_time))
                    start_time = time()
                    counter += 1
                    one_line = np.array(self.emg_data + self.acceleration_data + self.gyroscope_data + self.orientation_data)
                    if len(one_line) == 18:
                        self.batch.append(one_line)
                    if counter%10 == 0:
                        if key == 'esc':
                            sys.exit()
                        while len(self.batch) > 100:
                            self.batch.remove(self.batch[0])
                        if len(self.batch) > 99:
                            batch_list = list(self.batch)
                            np_batch = np.array(batch_list)
                            np_batch = np.reshape(np_batch, (1, np_batch.shape[0], np_batch.shape[1]))
                            prediction = int(self.model.predict(np_batch, verbose=0).argmax())
                            self.prev_predictions.append(prediction)
                            while len(self.prev_predictions) > 10:
                                del self.prev_predictions[0]
                            match prediction:
                                case 0:
                                    print("consume")
                                    if self.prev_predictions.count(prediction) >= 5:
                                        self.prev_predictions = []
                                        self.batch = []
                                        ui.display("consume")
                                case 1:
                                    print("sleep")
                                    if self.prev_predictions.count(prediction) >= 5:
                                        self.prev_predictions = []
                                        self.batch = []
                                        ui.display("sleep")
                                case 2:
                                    print("attention")
                                    if self.prev_predictions.count(prediction) >= 8:
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
    print("running system main")
    listener = EmgCollector(1, '100_set_model1.keras')
    listener.main()


if __name__ == '__main__':
    main()