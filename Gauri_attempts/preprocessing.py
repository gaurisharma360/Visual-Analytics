import os
import pandas as pd

base_path = r"D:\TUE Study Material\Q3\Visual Analytics\dataset"

folder_label_map = {
    "Z": "A",
    "O": "B",
    "N": "C",
    "F": "D",
    "S": "E"
}

all_rows = []

for folder, label in folder_label_map.items():
    folder_path = os.path.join(base_path, folder)

    print("Checking:", folder_path)
    if not os.path.exists(folder_path):
        print("Folder missing:", folder_path)
        continue

    files = sorted(os.listdir(folder_path))
    print(f"{folder} → {len(files)} files found")

    for file in files:
        if file.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r") as f:
                values = f.read().splitlines()

            values = [float(v.strip()) for v in values]

            filename_without_ext = os.path.splitext(file)[0]
            identifier = f"{label}.{filename_without_ext}"

            row = [identifier] + values + [label]
            all_rows.append(row)

columns = (
    ["ID"] +
    [f"X_{i}" for i in range(1, 4098)] +
    ["Y"]
)

df = pd.DataFrame(all_rows, columns=columns)
df.to_csv("bonn_eeg_combined.csv", index=False)

print("CSV file created successfully.")
