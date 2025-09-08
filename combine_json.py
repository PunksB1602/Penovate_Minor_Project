import json
import os

def combine_character_data(data_dir="imu_dataset", output_file="combined_dataset.json"):
    """
    Combine individual character JSON files into a single dataset with metadata.
    """
    combined_data = {
        "data": [],
        "metadata": {
            "num_samples": 0,
            "characters": set(),
            "samples_per_character": {}
        }
    }

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            character = filename[:-5]  # strip extension
            file_path = os.path.join(data_dir, filename)

            try:
                with open(file_path, "r") as f:
                    sequences = json.load(f)

                for seq in sequences:
                    combined_data["data"].append({
                        "character": character,
                        "sequence": seq,
                        "sequence_length": len(seq)
                    })

                combined_data["metadata"]["characters"].add(character)
                combined_data["metadata"]["samples_per_character"][character] = len(sequences)
                combined_data["metadata"]["num_samples"] += len(sequences)

                print(f"{character}: {len(sequences)} samples")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    combined_data["metadata"]["characters"] = sorted(combined_data["metadata"]["characters"])

    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"\nSaved combined dataset → {output_file}")
    print(f"Total samples: {combined_data['metadata']['num_samples']}")

    return combined_data


if __name__ == "__main__":
    combine_character_data()
