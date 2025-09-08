import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_training_data(json_file="combined_dataset.json", test_size=0.2, random_state=42, save_dir="./"):
    """
    Load IMU dataset from JSON, pad sequences, encode labels,
    split into train/test sets, and save to .npy files.
    """
    with open(json_file, "r") as f:
        dataset = json.load(f)

    sequences, labels = [], []
    for sample in dataset["data"]:
        sequences.append(sample["sequence"])
        labels.append(sample["character"])

    max_length = max(len(seq) for seq in sequences)

    X = []
    for seq in sequences:
        padded = seq + [[0.0] * len(seq[0])] * (max_length - len(seq))
        X.append(padded)
    X = np.array(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    np.save(f"{save_dir}/X_train.npy", X_train)
    np.save(f"{save_dir}/X_test.npy", X_test)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/y_test.npy", y_test)
    np.save(f"{save_dir}/label_encoder.npy", label_encoder.classes_)

    print("\nDataset Info")
    print(f"Features per timestep: {X.shape[2]}")
    print(f"Max sequence length: {max_length}")
    print(f"Classes: {label_encoder.classes_}")

    print("\nShapes")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    print("\nClass distribution")
    for idx, label in enumerate(label_encoder.classes_):
        train_count = np.sum(y_train == idx)
        test_count = np.sum(y_test == idx)
        print(f"{label}: train={train_count}, test={test_count}")

    return X_train, X_test, y_train, y_test, label_encoder


if __name__ == "__main__":
    prepare_training_data()
