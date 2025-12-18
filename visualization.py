import torch
import numpy as np
from sklearn.metrics import  roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from captum.attr import Occlusion, GradientShap
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


class AttributionVisualizer:

    def __init__(self, net, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.device = device
        self.input = input.unsqueeze(0)
        self.input.requires_grad = True

    def _convert_permute(self, algorithm):
        np_attr = algorithm.squeeze().cpu().detach().numpy()
        if np_attr.ndim == 2:
            return np_attr
        elif np_attr.ndim == 3:
            return np.transpose(np_attr, (1, 2, 0))

    def _original_image(self):
        img_tensor = self.input.squeeze().cpu().detach()
    
        mean = torch.tensor([0.456, 0.456, 0.456]).view(3,1,1)
        std = torch.tensor([0.224, 0.224, 0.224]).view(3,1,1)
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = np.transpose(img_denorm.numpy(), (1, 2, 0))
        return img_np

    def Occlusion_(self, target, strides=(3, 8, 8), sliding_window_shapes=(3, 15, 15)):
        baseline = torch.zeros_like(self.input)
        occlusion = Occlusion(self.net)
        attributions_occ = occlusion.attribute(self.input,strides=strides,
                                                target=target,
                                                sliding_window_shapes=sliding_window_shapes,
                                                baselines=baseline)
        return self._convert_permute(attributions_occ)

    def GradientShap_(self, target, n_samples=50, stdevs=0.0001):
        baseline = torch.zeros_like(self.input)
        gradient_shap = GradientShap(self.net)
        attributions_gs = gradient_shap.attribute(self.input,
                                          n_samples=n_samples,
                                          stdevs=stdevs,
                                          target=target, baselines=baseline)
        return self._convert_permute(attributions_gs)

    def viz_attr(self, index, save_dir):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        original_img = self._original_image()
        print(f"Original image shape: {original_img.shape}, min: {original_img.min()}, max: {original_img.max()}, dtype: {original_img.dtype}")
        axs[0].imshow(original_img)
        axs[0].axis("off")
        axs[0].set_title(f"Original Image")
        _ = viz.visualize_image_attr(attr=self.GradientShap_(target=0,n_samples=50, stdevs=0.0001),
                                     original_image=original_img, method="heat_map",
                                     show_colorbar=True, sign="all", title="Gradient Shap", # sign argument modified from "all" to "positive"
                                     plt_fig_axis=(fig, axs[1]))
        _ = viz.visualize_image_attr(attr=self.Occlusion_(target=0, strides=(3, 8, 8),
                                     sliding_window_shapes=(3, 15, 15)),
                                     original_image=original_img, method="heat_map",
                                     show_colorbar=True, sign="all", title="Occlusion", # sign argument modified from "all" to "positive"
                                     plt_fig_axis=(fig, axs[2]))
        axs[1].axis("off")
        axs[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_img_{index+1}.png")


def create_samples(loader, net, device):
    correct = 0
    total = 0
    TP = []
    TN = []
    FP = []
    FN = []
    all_labels = []
    pred_proba = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = torch.sigmoid(net(images))
            pred_proba.extend(outputs.cpu().numpy().ravel())
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy().ravel())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            for i, pred in enumerate(predicted):
                if pred == 1 and labels[i] == 1:
                    TP.append(images[i])
                elif pred == 0 and labels[i] == 0:
                    TN.append(images[i])
                elif pred == 1 and labels[i] == 0:
                    FP.append(images[i])
                elif pred == 0 and labels[i] == 1:
                    FN.append(images[i])
    y_true = all_labels
    tp_lst = TP[:10] if len(TP) > 10 else TP
    tn_lst = TN[:10] if len(TN) > 10 else TN
    fp_lst = FP[:10] if len(FP) > 10 else FP
    fn_lst = FN[:10] if len(FN) > 10 else FN

    return tp_lst, tn_lst, fp_lst, fn_lst, y_true, pred_proba

def plot_metrics(y_true, pred_proba_dict, path):
    fig = plt.figure(figsize=(8, 8))
    fig.set_dpi(300)
    for model_name, model_pred_proba in pred_proba_dict.items():
        fpr, tpr, thresholds = roc_curve(y_true, model_pred_proba, drop_intermediate=True)
        auc = roc_auc_score(y_true, model_pred_proba)
        plt.plot(fpr, tpr, label=f"{model_name}(AUC: {auc:.4f})")

    plt.title('ROC Curve', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

    plt.plot([0, 1], [0, 1], linestyle='--', color='r')
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./{path}/ROC_PR_Plots.png")
    
def plot_line_chart(model_list, accuracy_list, f1_score_list, exp_nos=4):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D']  
    colors = ['b', 'g', 'r']
    experiments = np.arange(1,exp_nos+1)
    for i, model in enumerate(model_list):
        plt.plot(experiments, accuracy_list[i], marker=markers[i], color=colors[i], linestyle='-', label=f'{model} Accuracy')
        plt.plot(experiments, f1_score_list[i], marker=markers[i], color=colors[i], linestyle='--', label=f'{model} F1 Score')
    plt.xlabel('Experiment Number')
    plt.ylabel('Score')
    plt.title('Model Performance Across Experiments')
    plt.xticks(experiments)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig("./Performance_metrics/GroupedPlot.png", dpi=300)
    plt.show()

def grouped_barplot(model_list, sensitivity_list, specificity_list, exp_nos=4):
    n_experiments = exp_nos
    experiments = ['Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']
    x = np.arange(len(models))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.bar(x - width/2, sensitivity_list[i], width, label='Sensitivity')
        ax.bar(x + width/2, specificity_list[i], width, label='Specificity')
        ax.set_title(experiments[i])
        ax.set_xticks(x)
        ax.set_xticklabels(model_list)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical')
    fig.legend(['Sensitivity', 'Specificity'],
               loc='upper center', ncol=2, frameon=False)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])
    plt.grid(True)
    plt.savefig("./Performance_metrics/GroupedBarPlot.png", dpi=300)
    plt.show()


