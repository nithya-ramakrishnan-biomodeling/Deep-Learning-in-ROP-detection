from train_n_eval import TrainEval
from models import CustomCNN, ResNet18_, ResNet18_pretrained, EfficientNet_pretrained, EfficientNet_
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os
from visualization import *
import warnings
warnings.filterwarnings("ignore")

# Function for visualizing the attibution maps of the model   
def _run_visualizer(model, model_path, image_lst, device, type):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    save_dir = f"./Visualization_Results/{model_path.split('/')[-1].split('.')[0]}/" + f"{type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        for index, image in enumerate(image_lst):
                av = AttributionVisualizer(model, image)
                av.viz_attr(index, save_dir)
                
 
 # main function to run the experiments               
def run(model_dict):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 60
    criterion = nn.BCEWithLogitsLoss()
    pred_proba_dict = {}
    
    for (model, pth_filename) in model_dict.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)    
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
        if not os.path.exists("./Trained_Models/"):
            os.makedir("./Trained_Models/")
        model_path = f"./Trained_Models/{pth_filename}"
        train_eval = TrainEval(model, train_dir, test_dir, model_path, num_epochs)
        train_eval.train_model(optimizer, criterion, scheduler)
        train_eval.evaluate_model()
        
        tp_lst, tn_lst, fp_lst, fn_lst, y_true, pred_proba = create_samples(train_eval.test_loader, model, device)
        pred_proba_dict[pth_filename.split('.')[0]] = pred_proba
        metrics_dir = "./Performance_Metrics/" + f"{pth_filename.split('_')[0]}/"
         
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            
        # Visualize samples
        _run_visualizer(model, model_path, tp_lst, device, "TP")
        _run_visualizer(model, model_path, tn_lst, device, "TN")
        _run_visualizer(model, model_path, fp_lst, device, "FP")      
        _run_visualizer(model, model_path, fn_lst, device, "FN")
        
    plot_metrics(y_true, pred_proba_dict, metrics_dir)

        
    
 
# Experiment 1 - Train on vittala dataset and test on vittala dataset
train_dir = "./Data/Vittala_Dataset"
test_dir = "./Data/Vittala_Dataset"
models = { CustomCNN():"Exp1_CustomCNN.pth",
          ResNet18_():"Exp1_ResNet18.pth",
          ResNet18_pretrained():"Exp1_ResNet18_pretrained.pth",
          EfficientNet_pretrained():"Exp1_EfficientNet_pretrained.pth",
          EfficientNet_():"Exp1_EfficientNet.pth"
        }

run(models)


# Experiment 2 - Train on opensource dataset and test on opensource dataset
train_dir = "./Data/Kaggle_Dataset"
test_dir = "./Data/Kaggle_Dataset"
models = { CustomCNN():"Exp2_CustomCNN.pth",
          ResNet18_():"Exp2_ResNet18.pth",
          ResNet18_pretrained():"Exp2_ResNet18_pretrained.pth",
          EfficientNet_pretrained():"Exp2_EfficientNet_pretrained.pth",
          EfficientNet_():"Exp2_EfficientNet.pth"
        }

run(models)

# Experiment 3 - Train on vittala dataset and test on opensource dataset

train_dir = "./Data/Vittala_Dataset"
test_dir = "./Data/Kaggle_Dataset"
models = { CustomCNN():"Exp3_CustomCNN.pth",
          ResNet18_():"Exp3_ResNet18.pth",
          ResNet18_pretrained():"Exp3_ResNet18_pretrained.pth",
          EfficientNet_pretrained():"Exp3_EfficientNet_pretrained.pth",
          EfficientNet_():"Exp3_EfficientNet.pth"
        }
run(models)
    
# Experiment 4 - Train on opensource dataset and test on vittala dataset

train_dir = "./Data/Kaggle_Dataset"
test_dir = "./Data/Vittala_Dataset"
models = { CustomCNN():"Exp4_CustomCNN.pth",
          ResNet18_():"Exp4_ResNet18.pth",
          ResNet18_pretrained():"Exp4_ResNet18_pretrained.pth",
          EfficientNet_pretrained():"Exp4_EfficientNet_pretrained.pth",
          EfficientNet_():"Exp4_EfficientNet.pth"
         }

run(models)
    
# Experiment 5 - Train on mixed dataset and test on mixed dataset
train_dir = "./Data/Mixed_Dataset"
test_dir = "./Data/Mixed_Dataset"
models = { CustomCNN():"Exp5_CustomCNN.pth",
          ResNet18_():"Exp5_ResNet18.pth",
          ResNet18_pretrained():"Exp5_ResNet18_pretrained.pth",
          EfficientNet_pretrained():"Exp5_EfficientNet_pretrained.pth",
          EfficientNet_():"Exp5_EfficientNet.pth"
         }

run(models)