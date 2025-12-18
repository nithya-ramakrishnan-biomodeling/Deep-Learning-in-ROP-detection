import os
import torch
from torch.utils.data import DataLoader, random_split
from load_data import AugmentedDataset, TestDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import subprocess
import random
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix


class TrainEval:

    def __init__(self, model, train_dir, test_dir, model_path, pth_filename, n_epochs, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model = model
        self.batch_size = batch_size
        self.model_path = model_path
        self.num_epochs = n_epochs
        self.test_loader = None
        self.checkpoint_path = f"./checkpoint/{pth_filename.split('.')[0]}checkpoint.pth"
        self.train_loss = []
        self.val_loss = []
        

    def _split_data(self, dataset):
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        return train_data, val_data    
        
    def _create_dataloader(self, augment):
        dataset = AugmentedDataset(self.train_dir, seed=0, augment=augment)
        test_data = TestDataset(self.test_dir, seed=0)
        train_data, val_data = self._split_data(dataset)
        train_loader = DataLoader(train_data, num_workers=4, pin_memory=True, prefetch_factor=4,
                                  batch_size=self.batch_size, shuffle=False) 
        val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=4, pin_memory=True,
                                shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False)
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def save_checkpoint(self, optimizer, scheduler, scaler, epoch):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "loss_history": (self.train_loss, self.val_loss)
        }
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(checkpoint, self.checkpoint_path)


    def load_checkpoint(self, optimizer, scheduler, scaler):
        if torch.cuda.is_available():
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint.get('loss_history', ([], []))
        return optimizer, scheduler, scaler, start_epoch, loss_history



    def train_model(self, optimizer, criterion, scheduler, augment):
        self.epochs_completed = self.num_epochs
        # Initialize GradScaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        if os.path.exists(self.model_path):
            if torch.cuda.is_available():
                model_weights = torch.load(self.model_path)
            else:
                model_weights = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(model_weights)
            print(f"Model loaded from {self.model_path}")
            return

        try:
            optimizer, scheduler, scaler, start_epoch, loss_history = self.load_checkpoint(optimizer, scheduler, scaler)
            self.train_loss, self.val_loss = loss_history
            if start_epoch < self.num_epochs:
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print(f"Training finished for {self.num_epochs} epochs")
        except FileNotFoundError:
            start_epoch = 1
            print("Started training from scratch")
            self.train_loss, self.val_loss = [], []

        train_loader, val_loader, _ = self._create_dataloader(augment=augment)

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()
            training_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                optimizer.zero_grad()

                # For mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                training_loss += loss.item()

            avg_train_loss = training_loss / len(train_loader)
            print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {avg_train_loss:.4f}")
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
            print(f"Validation Accuracy for the Epoch {epoch}: {100 * correct / total:.2f}%")
            self.save_checkpoint(optimizer, scheduler, scaler, epoch)
    
            # Save checkpoint and best model weights
            self.save_checkpoint(optimizer, scheduler, scaler, epoch)
    
        self.plot_loss_curve()

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def evaluate_model(self, exp_no):
        test_loader = self._create_dataloader(augment=False)[-1]
        if torch.cuda.is_available():
            model_weights = torch.load(self.model_path)
        else:
            model_weights = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_weights)
        self.model.eval()
        correct, total = 0, 0
        all_labels, all_preds, all_out_proba = [], [], []
        torch.cuda.empty_cache()
        with torch.no_grad():
            if not os.path.exists(f"./ROC_Curves/Exp{exp_no}"):
                os.makedirs(f"./ROC_Curves/Exp{exp_no}")
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                outputs = torch.sigmoid(self.model(images))
                all_labels.extend(labels.cpu().numpy())
                all_out_proba.extend(outputs.cpu().numpy())

        # Convert lists to arrays for vectorized operations
        all_labels = np.array(all_labels).flatten()
        all_out_proba = np.array(all_out_proba).flatten()

        best_threshold = self.plot_roc_curve(
            all_labels, all_out_proba,
            f"./ROC_Curves/Exp{exp_no}/{self.model_path.split('/')[-1].removesuffix('.pth')}_roc_curve.png"
        )

        all_predicted = (all_out_proba > best_threshold).astype(float)
        correct = np.sum(all_predicted == all_labels)
        total = len(all_labels)
        test_accuracy = 100 * correct / total
        test_f1 = f1_score(all_labels, all_predicted, zero_division=1)
        conf_mat = confusion_matrix(all_labels, all_predicted)
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")
        print(f"Final Test F1-Score: {test_f1:.4f}")
        print(f"Confusion Matrix:\n {conf_mat}")
        display = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Negative", "Positive"])
        display.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title("Confusion Matrix")
        if not os.path.exists(f"./Confusion_Matrix/Exp{exp_no}"):
            os.makedirs(f"./Confusion_Matrix/Exp{exp_no}")
        plt.savefig(f"./Confusion_Matrix/Exp{exp_no}/{self.model_path.split('/')[-1].removesuffix('.pth')}_confusion_matrix.png")
        specificity = conf_mat[0][0]/conf_mat.sum(axis=1)[0]
        sensitivity = conf_mat[1][1]/conf_mat.sum(axis=1)[1]
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        return test_accuracy, test_f1, specificity, sensitivity

    def plot_loss_curve(self):
        if not os.path.exists("./Loss_Curves"):
            os.makedirs("./Loss_Curves")
        epochs = [i for i in range(1, self.epochs_completed + 1)]
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(epochs, self.train_loss, label="Train Loss", linewidth=2)
        plt.plot(epochs, self.val_loss, label="Validation Loss", linewidth=2)
        plt.title("Loss Curve", fontsize=16)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if not os.path.exists(f"./Loss_Curves/"
                              f"{self.model_path.split('/')[-1].removesuffix('.pth')}_loss_curve.png"):
            plt.savefig(f"./Loss_Curves/{self.model_path.split('/')[-1].removesuffix('.pth')}_loss_curve.png")
        
    def plot_roc_curve(self, y_true, y_pred_proba, save_pth):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        J = tpr - fpr
        best_idx = J.argmax()
        best_threshold = thresholds[best_idx]
        
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', label=f'Best Threshold = {best_threshold:.2f}')
        plt.annotate(f'{best_threshold:.2f}', (fpr[best_idx], tpr[best_idx]), textcoords="offset points", xytext=(10,-10), ha='center', color='red')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc='lower right')
        plt.savefig(save_pth, dpi=300)
        plt.close()
        return best_threshold
        
