# Deep Learning for prediction of Retinopathy of prematurity (ROP) in preterm infants from fundus images

## Description
Retinopathy of prematurity (ROP) is a retinal disease affecting preterm infants, causing abnormal development of blood vessels. If not treated, this condition might cause partial to complete vision loss.
For early detection and automation, fundus images are fed to deep neural networks and the presence of disease is predicted.
Here we have tried and built a generalized deep learning model which is able to detect the presence of the disease from an out of the distribution(OOD) image set.

## Dataset info
Publicly available dataset is used for training the model, and it is tested on a private dataset for validation of the performace of the model.
For training, only images upto stage 3 were combined and used along with negative class. Class balance was ensured before passing to the model for training.
link to training dataset [https://www.kaggle.com/datasets/jananowakova/retinal-image-dataset-of-infants-and-rop](https://www.kaggle.com/datasets/jananowakova/retinal-image-dataset-of-infants-and-rop)

## Directory structure
The datafiles are to be stored in the below directory structure inside the parent directory.

<pre> ``` Data/ ├── dataset1/ │ ├── Positive/ │ └── Negative/ ├── dataset2/ │ ├── Positive/ │ └── Negative/ ``` </pre>
    
## Model info
A total of 5 models were defined and trained on the above dataset. The performance of each model is compared and evaluated using the private dataset.
Captum package was used for the explainability of the models. Link to the documentation: [https://captum.ai/api/attribution.html](https://captum.ai/api/attribution.html)

## Reproducibility of code
Clone this repository, initialize a virtual environment and run the following command for installing the dependencies.
### pip install -r requirements.txt





