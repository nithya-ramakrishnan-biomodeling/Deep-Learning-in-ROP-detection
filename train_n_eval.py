import os
import torch
from torch.utils.data import DataLoader, random_split
from load_data import AugmentedDataset, TestDataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix


class TrainEval:
    
    def __init__(self, model, train_dir, test_dir, model_path, n_epochs, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model = model
        self.batch_size = batch_size
        self.model_path = model_path
        self.num_epochs = n_epochs
        self.test_loader = None
        self.train_loss = []     
        self.val_loss = []
        
    def _split_data(self, dataset):
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size 
        train_data, val_data = random_split(dataset, [train_size, val_size])  
        return train_data, val_data

    def _create_dataloader(self):
        dataset = AugmentedDataset(self.train_dir)
        test_data = TestDataset(self.test_dir)
        train_data, val_data = self._split_data(dataset)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader
        
    def train_model(self, optimizer, criterion, scheduler):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Model loaded from {self.model_path}")
        else:
            print("Training started...")
            train_loader, val_loader = self._create_dataloader()[:2]
            for epoch in range(self.num_epochs):
                self.model.train()
                training_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()

                avg_train_loss = training_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}")
                scheduler.step(avg_train_loss) 

                self.train_loss.append(avg_train_loss)

                self.model.eval()
                correct, total = 0, 0
                validation_loss = 0.0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                        outputs = torch.sigmoid(self.model(images))  
                        predicted = (outputs > 0.5).float()
                        loss = criterion(outputs, labels)
                        validation_loss += loss.item()
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                        
                avg_val_loss = validation_loss / len(val_loader)
                self.val_loss.append(avg_val_loss)
                
                print(f"Validation Accuracy for the Epoch {epoch+1}: {100 * correct / total:.2f}%")
                
            torch.save(self.model.state_dict(), self.model_path)
            self.plot_loss_curve()
            print(f"Model saved to {self.model_path}")
            
    def evaluate_model(self):
        test_loader = self._create_dataloader()[-1]
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        correct, total = 0, 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                outputs = torch.sigmoid(self.model(images))  
                predicted = (outputs > 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
    
        test_accuracy = 100 * correct / total
        test_f1 = f1_score(all_labels, all_preds, zero_division=1)  
        conf_mat = confusion_matrix(all_labels, all_preds)
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")
        print(f"Final Test F1-Score: {test_f1:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
        specificity = conf_mat[0][0]/conf_mat.sum(axis=1)[0]
        sensitivity = conf_mat[1][1]/conf_mat.sum(axis=1)[1]
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
    
    def plot_loss_curve(self):
        if not os.path.exists("./Loss_Curves"):
            os.makedirs("./Loss_Curves")
        epochs = [i for i in range(1, self.num_epochs + 1)]
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(epochs, self.train_loss, label="Train Loss", linewidth=2)
        plt.plot(epochs, self.val_loss, label="Validation Loss", linewidth=2)
        plt.title("Loss Curve", fontsize=16)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if not os.path.exists(f"./Loss_Curves/{self.model_path.split("/")[-1].removesuffix(".pth")}_loss_curve.png"):
            plt.savefig(f"./Loss_Curves/{self.model_path.split("/")[-1].removesuffix(".pth")}_loss_curve.png")
    
    
        
