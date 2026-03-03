import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, num_layers=1, dropout=0.3, padding_idx=0):
        super(EncoderLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout if num_layers > 1 else 0)
        self.out_dim = hidden_size

    def forward(self, questions):
        embedded = self.embedding(questions)          
        _, (hidden, _) = self.lstm(embedded)         
        return hidden[-1]    

class VQAModel(nn.Module):
    def __init__(self, image_encoder, question_encoder, fusion_hidden=512, dropout=0.3):
        super(VQAModel, self).__init__()
        
        self.image_encoder = image_encoder
        self.question_encoder = question_encoder
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.linear.out_features + self.question_encoder.out_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)  
        )

    def forward(self, images, questions):

        img_feat = self.image_encoder(images)       
        q_feat = self.question_encoder(questions)  
        combined = torch.cat((img_feat, q_feat), dim=1)  
        output = self.classifier(combined)   
                    
        return output
