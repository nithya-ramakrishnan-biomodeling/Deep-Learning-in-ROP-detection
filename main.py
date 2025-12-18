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


def counter():
    count = 0
    while True:
        count += 1
        yield count

def _run_visualizer(exp_no, model, model_path, image_lst, device, type):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    save_dir = f"./Visualization_Results/Exp{exp_no}/{model_path.split('/')[-1].split('.')[0]}/" + f"{type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        for index, image in enumerate(image_lst):
                image = image.to(device)
                av = AttributionVisualizer(model, image)
                av.viz_attr(index, save_dir)
        


 # main function to run the experiments
def run(model_dict, counter, train_dir, test_dir, augment=False):
    exp_no = next(counter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    criterion = nn.BCEWithLogitsLoss()
    pred_proba_dict = {}
    accuracy_list = []
    f1_score_list = []
    sensitivity_list = []
    specificity_list = []

    for (index,(model, pth_filename)) in enumerate(model_dict.items()):
        model = model.to(device) # Move model to the device
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
        if not os.path.exists("./Trained_Models/"):
            os.makedirs("./Trained_Models/")
        model_path = f"./Trained_Models/{pth_filename}"
        train_eval = TrainEval(model, train_dir, test_dir, model_path, pth_filename, num_epochs)
        train_eval.train_model(optimizer, criterion, scheduler, augment)
        test_accuracy, test_f1, test_sensitivity, test_specificity = train_eval.evaluate_model(exp_no)
        accuracy_list.append(test_accuracy)
        f1_score_list.append(test_f1)
        sensitivity_list.append(test_sensitivity)
        specificity_list.append(test_specificity)
        metrics_dir = f"./Performance_metrics/Exp{exp_no}"
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        tp_lst, tn_lst, fp_lst, fn_lst, y_true, pred_proba = create_samples(train_eval.test_loader, model, device)
        pred_proba_dict[pth_filename.split('.')[0]] = pred_proba
        # if exp_no > 2:
        # Visualize samples
        _run_visualizer(exp_no, model, model_path, tp_lst, device, "TP")
        _run_visualizer(exp_no, model, model_path, tn_lst, device, "TN")
        _run_visualizer(exp_no, model, model_path, fp_lst, device, "FP")
        _run_visualizer(exp_no, model, model_path, fn_lst, device, "FN")

    plot_metrics(y_true, pred_proba_dict, metrics_dir)
    return accuracy_list, f1_score_list, sensitivity_list, specificity_list


def main():
    cntr = counter()
    
    f1_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    dir1 = "/kaggle/input/vittala-dataset/Vittala_Dataset"
    dir2 = "/kaggle/input/kaggle-dataset/Kaggle_Dataset"
    
    # Experiment 1 - Train on vittala dataset and test on vittala dataset

    models = { 
              CustomCNN():"V_customCNN.pth",
              ResNet18_pretrained():"V_ResNet18_pretrained.pth",
              EfficientNetB0_pretrained():"V_EfficientNetB0_pretrained.pth"
           }
    
    accuracy1, f1_score1, sensitivity1, specificity1 = run(models, cntr, dir1, dir2, augment=False)
    accuracy_list.append(accuracy1)
    f1_list.append(f1_score1)
    sensitivity_list.append(sensitivity1)
    specificity_list.append(specificity1)
    
    # Experiment 2 - Train on opensource dataset and test on opensource dataset
    models = { 
              CustomCNN():"K_customCNN.pth",
              ResNet18_pretrained():"K_ResNet18_pretrained.pth",
              EfficientNetB0_pretrained():"K_EfficientNetB0_pretrained.pth"
            }
    
    accuracy2, f1_score2, sensitivity2, specificity2 = run(models, cntr, dir2, dir1, augment=False)
    accuracy_list.append(accuracy2)
    f1_list.append(f1_score2)
    sensitivity_list.append(sensitivity2)
    specificity_list.append(specificity2)
    
    # Experiment 3 - Train on vittala dataset and test on opensource dataset
    models = { 
              CustomCNN():"V_Augmented_customCNN.pth",
              ResNet18_pretrained():"V_Augmented_ResNet18_pretrained.pth",
              EfficientNetB0_pretrained():"V_Augmented_EfficientNetB0_pretrained.pth"
            }
    accuracy3, f1_score3,sensitivity3, specificity3 = run(models, cntr, dir1, dir2, augment=True)
    accuracy_list.append(accuracy3)
    f1_list.append(f1_score3)
    sensitivity_list.append(sensitivity3)
    specificity_list.append(specificity3)
        
    # Experiment 4 - Train on opensource dataset and test on vittala dataset
    models = { 
              CustomCNN():"K_Augmented_customCNN.pth",
              ResNet18_pretrained():"K_Augmented_ResNet18_pretrained.pth",
              EfficientNetB0_pretrained():"K_Augmented_EfficientNetB0_pretrained.pth",
             }
    
    accuracy4, f1_score4, sensitivity4, specificity4 = run(models, cntr, dir2, dir1, augment=True)
    accuracy_list.append(accuracy4)
    f1_list.append(f1_score4)
    sensitivity_list.append(sensitivity4)
    specificity_list.append(specificity4)


    accuracy_list = np.array(accuracy_list).T/100.0
    f1_score_list = np.array(f1_list).T
    model_list = ["CustomCNN", "ResNet18", "EfficientNetB0"]
    plot_line_chart(model_list, accuracy_list, f1_score_list)
    grouped_barplot(model_list, sensitivity_list, specificity_list)

if __name__ == "__main__":
    main()


