import serial
import numpy as np
from scipy.signal import butter, filtfilt
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from threading import Thread

class IMUPredictorGUI:
    def __init__(self, model_path, label_encoder_path, port='COM6', baud_rate=115200):
        """GUI for real-time IMU character prediction"""
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = np.load(label_encoder_path, allow_pickle=True)
        self.serial = serial.Serial(port, baud_rate)

        self.filter_order = 2
        self.cutoff_freq = 20
        self.sampling_rate = 100

        self.root = tk.Tk()
        self.root.title("IMU Character Predictor")
        self.root.configure(bg="white")

        self.text_display = tk.Text(
            self.root,
            font=("Helvetica", 60),
            bg="white",
            fg="black",
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.text_display.pack(expand=True, fill="both")

        button_frame = tk.Frame(self.root, bg="white")
        button_frame.pack(fill="x", pady=10)

        clear_button = tk.Button(button_frame, text="Clear", font=("Helvetica", 20), command=self.clear_text)
        clear_button.pack(side=tk.LEFT, padx=20)

        save_button = tk.Button(button_frame, text="Save", font=("Helvetica", 20), command=self.save_text)
        save_button.pack(side=tk.RIGHT, padx=20)

        self.running = True
        self.thread = Thread(target=self.collect_and_predict)
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def butter_lowpass(self):
        nyquist = 0.5 * self.sampling_rate
        cutoff = self.cutoff_freq / nyquist
        return butter(self.filter_order, cutoff, btype="low")

    def apply_lowpass_filter(self, data):
        b, a = self.butter_lowpass()
        data_array = np.array(data)
        filtered = np.zeros_like(data_array)
        for i in range(data_array.shape[1]):
            filtered[:, i] = filtfilt(b, a, data_array[:, i])
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

    def preprocess_sequence(self, sequence):
        if not sequence:
            return None
        filtered = self.apply_lowpass_filter(sequence)
        rel_motion = self.calculate_relative_motion(filtered)
        return self.normalize_data(rel_motion)

    def append_prediction(self, text):
        self.text_display.config(state=tk.NORMAL)
        self.text_display.insert(tk.END, text)
        self.text_display.config(state=tk.DISABLED)

    def clear_text(self):
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)
        self.text_display.config(state=tk.DISABLED)

    def save_text(self):
        text = self.text_display.get(1.0, tk.END).strip()
        if text:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if path:
                with open(path, "w") as f:
                    f.write(text)

    def collect_and_predict(self):
        sequence, recording = [], False
        while self.running:
            if self.serial.in_waiting:
                line = self.serial.readline().decode("utf-8").strip()
                if line == "START":
                    recording, sequence = True, []
                elif line == "END":
                    if recording and sequence:
                        processed = self.preprocess_sequence(sequence)
                        if processed is not None:
                            model_input = np.expand_dims(processed, axis=0)
                            pred = self.model.predict(model_input, verbose=0)
                            idx = np.argmax(pred[0])
                            char = self.label_encoder[idx]
                            self.root.after(0, self.append_prediction, char.lower())
                    recording = False
                elif recording:
                    try:
                        values = [float(x) for x in line.split(",")]
                        if len(values) == 12:
                            sequence.append(values)
                    except ValueError:
                        continue

    def on_close(self):
        self.running = False
        self.serial.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    MODEL_PATH = "best_model.keras"
    LABEL_ENCODER_PATH = "label_encoder.npy"
    app = IMUPredictorGUI(MODEL_PATH, LABEL_ENCODER_PATH)
    app.run()


if __name__ == "__main__":
    main()
