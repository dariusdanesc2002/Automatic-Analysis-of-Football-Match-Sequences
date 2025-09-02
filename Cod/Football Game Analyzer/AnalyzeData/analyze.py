import zipfile
import os
import yaml
from collections import Counter
import matplotlib.pyplot as plt


class Analyze:
    zip_path = ""
    extract_dir = ""
    class_distribution = []

    def __init__(self):

        zip_path = "C:/Users/dariu/Downloads/football-players-detection.v12i.yolov8.zip"
        extract_dir = "C:/Users/dariu/OneDrive/Desktop/Licenta/Cod/Football Game Analyzer/AnalyzeData/ExtractedZip"

    def function(self):
        zip_path = "C:/Users/dariu/Downloads/football-players-detection.v12i.yolov8.zip"
        extract_dir = "/AnalyzeData/ExtractedZip"

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        os.listdir(extract_dir)

        train_labels_dir = os.path.join(extract_dir, "train", "labels")
        label_files = os.listdir(train_labels_dir)

        class_counts = Counter()

        for label_file in label_files:
            with open(os.path.join(train_labels_dir, label_file), "r") as f:
                for line in f:
                    class_id = line.strip().split()[0]
                    class_counts[class_id] += 1

        with open(os.path.join(extract_dir, "data.yaml"), "r") as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml["names"]

        self.class_distribution = {class_names[int(k)]: v for k, v in class_counts.items()}

    def plot(self):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.class_distribution.keys(), self.class_distribution.values(),
                       color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"])
        plt.title("Distribuția claselor în setul de antrenare", fontsize=16, weight='bold')
        plt.xlabel("Clasă", fontsize=12)
        plt.ylabel("Număr de instanțe", fontsize=12)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 50, f'{height}', ha='center', va='bottom',
                     fontsize=10)

        plt.tight_layout()
        fancy_plot_path = "C:/Users/dariu/OneDrive/Desktop/Licenta/Cod/Football Game Analyzer/AnalyzeData/img.png"
        plt.savefig(fancy_plot_path)
        plt.show()
