# **An Ensemble Deep Learning Model to Classify Scoliosis and Healthy Subjects**

## **Adolescent idiopathic scoliosis**

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Scoliosis_%2815-year-old%29.jpg" width="250">
  <br>
  (Image source <a href="https://commons.wikimedia.org/"> Wikimedia Commons</a>)
</p>

**Adolescent idiopathic scoliosis** (AIS) is a three-dimensional deformity of the spine, which is characterized by deformation of vertebral column curvatures on the sagittal, frontal and transverse plane. X-ray are used to diagnose AIS, as they allow to detect vertebral rotation and to compute Cobb angle, needed for AIS classification. X-rays, however, carry health risk from repetitive exposure to ionizing radiation and cannot aid the physician to detect postural changes associated to AIS.

Recently, **Video-Raster-Stereography** (VRS) has been proposed as an objective non-invasive method for instrumented three-dimensional (3D) back shape analysis and reconstruction of spinal curvatures and deformities without radiation exposure.

The main drawback with the application of VRS to clinical practice like in AIS screening, is represented by the lack of a codified system to analyze and interpret the whole number of parameters derived from any single acquisition.[[1]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0261511)

A narrative review performed by Chen et al. in 2021 [[2]](https://atm.amegroups.com/article/view/60113/html), describes the application of ML in clinical practice procedures regarding scoliosis in various medical phases. In particular, the authors emphasized that an accurate diagnosis with ML can help surgeons avoid misjudgment.

# **Ensemble Deep Learning**

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*MxD8Kn_Rn9p_Au4MOGgsmg.png">
  <br>
  (Image source <a href="https://towardsdatascience.com/neural-networks-ensemble-33f33bea7df3">Medium article</a>)
</p>

Ensemble modeling is the process by which a machine learning model combines distinct base models to generate generalized predictions using a combination of the predictive power of each of its components. Each base model differs with respect to the variable elements i.e. training data used and algorithm/model architecture. Each of these may capture only a few aspects or learn effectively from only a part of the training data, due to its specific tuning. Ensemble modeling provides us with the opportunity to combine all such models to acquire a single superior model which is based on learnings from most or all parts of the training dataset. Averaging of predictions also help eradicate the issue of individual models fixating on local minima.[[3]](https://www.analyticsvidhya.com/blog/2021/10/ensemble-modeling-for-neural-networks-using-large-datasets-simplified/)

# **This project**

In this project we constructed and trained an Ensemble Neural Network Model to classify scoliosis and healthy subjects. The model predicts the probability that a subject suffers from AIS basd on the VRS data. The Ensemble Neural Network performs quite well achieving a balanced accuracy over 85%.

## Dataset

We leveraged the data constituted by the Video-Raster-Stereography (VRS) measures of subjects who have undergone a clinical check and have been diagnosticated as healthy/AIS. Acquisition of data was performed through VRS by the Formetric™4D system (Diers International GmbH, Schlangenbad, Germany) for the research work by Colombo T et al. 2021 on [Supervised and unsupervised learning to classify scoliosis and healthy subjects based on non-invasive rasterstereography analysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0261511)

## Scripts

* ``ensemble.py``: contains the implementation of the Ensemble Deep Learning Model.

* ``train_ensemble.py``: trains the ensemble deep learning algorithm model on a dataset composed of 272 scoliotic subjects and 192 healthy subjects, and saves the weights and parameters of the model under the selected directory (in this case, the trained ensemble model is saved under ``model``) in the project folder. Then, it loads the model and assesses the quality of predictions using the ``balanced_accuracy_score``.

## Installation

Install Python (>=3.6):

```
    sudo apt-get update
    sudo apt-get install python3.6
```

Clone this repository:

```
git clone https://github.com/wiguider/Ensemble-Deep-Learning-to-Classify-Scoliosis-and-Healthy-Subjects.git
```

Install the requirements:

```
pip install -r requirements.txt
```
