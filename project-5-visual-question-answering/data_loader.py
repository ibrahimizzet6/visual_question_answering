import json
import os
from torch.utils.data import Dataset
from PIL import Image

class BinaryAbstractVQA(Dataset):
    def __init__(self, image_folder, question_path, annotation_path, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        with open(question_path, "r") as f:
            questions = json.load(f)["questions"]

        with open(annotation_path, "r") as f:
            annotations = json.load(f)["annotations"]

        answer_dict = {
            ann["question_id"]: ann["multiple_choice_answer"]
            for ann in annotations
        }

        self.samples = []

        for q in questions:
            qid = q["question_id"]
            img_id = q["image_id"]
            question_text = q["question"]

            if qid not in answer_dict:
                continue

            answer = answer_dict[qid]
            label = 1 if answer.lower() == "yes" else 0

            image_path = os.path.join(
                self.image_folder,
                f"abstract_v002_train2015_{img_id:012d}.png"
            )

            if os.path.exists(image_path):
                self.samples.append((image_path, question_text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, question, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, question, label