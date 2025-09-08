import serial
import numpy as np
from scipy.signal import butter, filtfilt
import tensorflow as tf

class IMUPredictor:
    def __init__(self, model_path, label_encoder_path, port="COM6", baud_rate=115200):
        """Load model, labels, and set up serial connection."""
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = np.load(label_encoder_path, allow_pickle=True)
        self.serial = serial.Serial(port, baud_rate)

        self.filter_order = 2
        self.cutoff_freq = 20  # Hz
        self.sampling_rate = 100  # Hz

    def butter_lowpass(self):
        nyquist = 0.5 * self.sampling_rate
        cutoff = self.cutoff_freq / nyquist
        return butter(self.filter_order, cutoff, btype="low", analog=False)

    def apply_lowpass_filter(self, data):
        b, a = self.butter_lowpass()
        arr = np.array(data)
        filtered = np.zeros_like(arr)
        for i in range(arr.shape[1]):
            filtered[:, i] = filtfilt(b, a, arr[:, i])
        return filtered.tolist()

    def calculate_relative_motion(self, data):
        arr = np.array(data)
        imu1, imu2 = arr[:, :6], arr[:, 6:]
        rel = imu1 - imu2
        return np.concatenate([imu1, imu2, rel], axis=1).tolist()

    def normalize_data(self, data):
        arr = np.array(data)
        mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
        std = np.where(std == 0, 1, std)
        return (arr - mean) / std

    def preprocess_sequence(self, seq):
        if not seq:
            return None
        seq = self.apply_lowpass_filter(seq)
        seq = self.calculate_relative_motion(seq)
        return self.normalize_data(seq)

    def collect_and_predict(self):
        sequence, recording = [], False
        print("Ready to predict. Write a character...")

        while True:
            if self.serial.in_waiting:
                line = self.serial.readline().decode("utf-8").strip()

                if line == "START":
                    recording, sequence = True, []
                    print("Recording started...")
                elif line == "END":
                    if recording and sequence:
                        print("Recording ended. Processing...")
                        seq = self.preprocess_sequence(sequence)
                        if seq is not None:
                            model_input = np.expand_dims(seq, axis=0)
                            pred = self.model.predict(model_input, verbose=0)[0]
                            idx, conf = np.argmax(pred), pred[np.argmax(pred)]
                            char = self.label_encoder[idx]
                            print(f"\nPredicted: {char.lower()}  (conf: {conf:.2f})")
                            print("Ready for next character...")
                    recording = False
                elif recording:
                    try:
                        vals = [float(x) for x in line.split(",")]
                        if len(vals) == 12:
                            sequence.append(vals)
                    except ValueError:
                        continue

    def run(self):
        print("IMU Predictor Started")
        print("Write characters to see predictions (Ctrl+C to stop)")
        try:
            self.collect_and_predict()
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.serial.close()
            print("Serial connection closed")


def main():
    MODEL_PATH = "best_model.keras"
    LABEL_ENCODER_PATH = "label_encoder.npy"
    predictor = IMUPredictor(MODEL_PATH, LABEL_ENCODER_PATH)
    predictor.run()


if __name__ == "__main__":
    main()
