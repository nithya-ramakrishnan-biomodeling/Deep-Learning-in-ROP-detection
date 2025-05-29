import torch
import numpy as np
from sklearn.metrics import  roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from captum.attr import Occlusion, GradientShap, LayerGradCam
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class AttributionVisualizer:
    
    def __init__(self, net, input):
        self.net = net
        self.input = input.unsqueeze(0)
        self.input.requires_grad = True
    
    def _convert_permute(self, algorithm):
        np_attr = algorithm.squeeze().cpu().detach().numpy()
        if np_attr.ndim == 2:
            return np_attr
        elif np_attr.ndim == 3:
            return np.transpose(np_attr, (1, 2, 0))
       
    def _original_image(self):
        return np.transpose((self.input.squeeze().cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)).astype(np.uint8)
    
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
        axs[0].imshow(original_img)
        axs[0].axis("off")
        axs[0].set_title(f"Original Image")
        _ = viz.visualize_image_attr(attr=self.GradientShap_(target=0,n_samples=50, stdevs=0.0001), 
                                     original_image=original_img, method="heat_map", 
                                     show_colorbar=True, sign="all", title="Gradient Shap", 
                                     plt_fig_axis=(fig, axs[1]))
        _ = viz.visualize_image_attr(attr=self.Occlusion_(target=0, strides=(3, 8, 8), 
                                     sliding_window_shapes=(3, 15, 15)),
                                     original_image=original_img, method="heat_map", 
                                     show_colorbar=True, sign="all", title="Occlusion", 
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


