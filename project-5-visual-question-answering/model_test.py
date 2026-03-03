import torch
from PIL import Image
from preprocess import image_transform, numericalize, pad_sequence, build_vocab
from model import EncoderCNN, EncoderLSTM, VQAModel
from main import vocab

embed_size = 512
vocab_size = len(vocab)  
cnn_encoder = EncoderCNN(embed_size=embed_size)
lstm_encoder = EncoderLSTM(vocab_size=vocab_size, hidden_size=embed_size)
model = VQAModel(cnn_encoder, lstm_encoder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load("vqa_binary_model.pth", map_location=device))
model.eval()


test_image_path = r"C:\Users\BB\python_icin\CV\Projeler\project-5-visual-question-answering\images\abstract_v002_train2015_000000000004.png"
test_question = "Are there two girls?"


image = Image.open(test_image_path).convert("RGB")
image = image_transform(image).unsqueeze(0).to(device)  


q_seq = numericalize(test_question, vocab)
q_seq = pad_sequence(q_seq, max_length=20)
question_tensor = torch.tensor(q_seq, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image, question_tensor)
    prob = torch.sigmoid(output).item() 

print(f"Predicted probability of 'yes': {prob:.4f}")
print("Predicted answer:", "Yes" if prob >= 0.5 else "No")