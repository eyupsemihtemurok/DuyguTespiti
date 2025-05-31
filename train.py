import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# --- Etiketler ---
etiketler = {
    0: "anger",
    1: "contempt",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprise"
}

# --- Dataset Sınıfı ---
class EmotionDataset(Dataset):
    def __init__(self, data_df, root_dir, transform=None):
        self.data = data_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['pth'])
        label = int(self.data.iloc[idx]['label'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    csv_path = r"D:\\DuyguTespiti/labels_numeric.csv"
    root_dir = r"D:\\DuyguTespiti"

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # --- Transformasyonlar ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # --- Dataset ve Sampler ---
    train_dataset = EmotionDataset(train_df, root_dir, transform=train_transform)
    val_dataset = EmotionDataset(val_df, root_dir, transform=val_transform)

    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    train_sample_weights = [class_weights[label] for label in train_df['label']]
    train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model = model.to(device)

    # --- Loss, Optimizer ve Scheduler ---
    # loss 
    criterion = nn.CrossEntropyLoss()
    # overfitting önleme
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # öğrenme oranını azaltma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- Eğitim ---
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

    # --- Model Kaydet ---
    torch.save(model.state_dict(), "duygu_modeli_resnet50.pth")
    print("✅ Model kaydedildi: duygu_modeli_resnet50.pth")

if __name__ == '__main__':
    main()