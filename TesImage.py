import torch
from PIL import Image 
from torch import nn, save, load 
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

#ambil data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )
     
    def forward(self, x):
        return self.model(x)
 
#Network loss Optimizer   
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

#Train Model
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))
        
    img = Image.open('img5.png')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    
    Output = f"Hasil Identifikasi : {torch.argmax(clf(img_tensor))}"
    print(Output)
    
    all_preds = []
    all_labels = []
    
    clf.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in dataset:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            preds = torch.argmax(yhat, dim=1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")