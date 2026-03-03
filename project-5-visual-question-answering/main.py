import os
import json
from PIL import Image as PILImage 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import BinaryAbstractVQA
from preprocess import image_transform, numericalize, pad_sequence, encode_answer, build_vocab
from model import EncoderCNN, EncoderLSTM, VQAModel


image_folder = r"C:\\Users\\BB\\python_icin\\CV\\Projeler\\project-5-visual-question-answering\\images" 
question_path = r"C:\\Users\\BB\\python_icin\\CV\\Projeler\\project-5-visual-question-answering\\OpenEnded_abstract_v002_train2017_questions.json"
annotation_path = r"C:\\Users\\BB\\python_icin\\CV\\Projeler\\project-5-visual-question-answering\abstract_v002_train2017_annotations.json"


with open(question_path, "r") as f:
        questions_data = json.load(f)["questions"]

questions_text = [q["question"] for q in questions_data]
vocab = build_vocab(questions_text, min_freq=1)

class VQADatasetWrapper(BinaryAbstractVQA):
    def __getitem__(self, idx):
        image, question, label = super().__getitem__(idx)
        
        
        q_seq = numericalize(question, vocab)
        q_seq = pad_sequence(q_seq, max_length=20)
        q_tensor = torch.tensor(q_seq, dtype=torch.long)
        
        
        label_tensor = torch.tensor(label, dtype=torch.float).unsqueeze(0)  
        
       
        if self.transform and isinstance(image,PILImage.Image):
            image = self.transform(image)

        return image, q_tensor, label_tensor

if __name__ == "__main__":
    
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    dataset = VQADatasetWrapper(image_folder, question_path, annotation_path, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    embed_size = 512
    cnn_encoder = EncoderCNN(embed_size=embed_size)
    lstm_encoder = EncoderLSTM(vocab_size=vocab_size, hidden_size=embed_size)
    model = VQAModel(cnn_encoder, lstm_encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    num_epochs = 10

    for epoch in range(num_epochs):
      print("epoch:", epoch+1)
      model.train()
      running_loss = 0.0
      for images, questions, labels in dataloader:
        images, questions, labels = images.to(device), questions.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, questions)  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

      epoch_loss = running_loss / len(dataloader.dataset)
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    save_path = "vqa_binary_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")