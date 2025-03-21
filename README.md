# CIFAR-10 Image Classification: Comparison of Vision Transformer vs. ResNet-18 vs. CNN-MLP Hybrid


## Overview  
This project explores **Vision Transformer (ViT), ResNet-18 (Transfer Learning), and a CNN-MLP Hybrid model** for **image classification on the CIFAR-10 dataset**. Each model is trained and evaluated separately to analyze performance in terms of **accuracy, training efficiency, memory usage, and inference speed**.  

- **Vision Transformer (ViT)** → Uses a Transformer-based approach to process images as **patch sequences**, leveraging self-attention mechanisms for classification.  
- **ResNet-18 (Transfer Learning)** → A **CNN-based model** with residual connections, pretrained on a large-scale dataset and fine-tuned for CIFAR-10 classification.  
- **CNN-MLP Hybrid** → A custom architecture where **CNN extracts image features**, which are then **fed into an MLP classifier** instead of using a Transformer.  

The project also includes a **Flask-based web application** that allows users to:  
- Select a classification model (ViT, ResNet, or CNN-MLP).  
- Upload one or more images for real-time classification.  
- View correct and incorrect classifications, along with predicted labels.  

The goal of this study is to compare **modern Transformer-based models with traditional CNN architectures**, analyzing their **trade-offs in accuracy, computational cost, and practical deployment**.  

---

## Key Objectives  
- **Preprocess & augment CIFAR-10 images** for training  
- **Implement and compare multiple deep learning architectures**:  
   - **Vision Transformer (ViT)** → Transformer-based image classification  
   - **ResNet-18 (Transfer Learning)** → CNN with pretrained feature extraction  
   - **CNN-MLP Hybrid** → CNN-based feature extraction + MLP classifier  
- **Tune hyperparameters** (layers, attention heads, learning rate, batch size, patch size)  
- **Evaluate models using classification metrics** (Accuracy, Precision, Recall, F1-score)  
- **Deploy trained models on a local machine** for real-time inference  
- **Develop a user-friendly frontend** where users can **upload single or multiple images** and classify them  
- **Visualize model predictions** for test images, highlighting **correct & incorrect classifications**  
- **Analyze computational efficiency** (training time, memory usage, inference speed)  

---

## Dataset Information  
- **Dataset Source**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
- **Total Images**: 60,000 images (32x32 pixels, 10 classes)  
- **Training Set**: 50,000 images  
- **Test Set**: 10,000 images  
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
  
---

## Repository Contents  
- `README.md` → Project documentation (**to be expanded with additional details**).  
- `i201819_ImamaAmjad_Ass3.pdf` → Detailed **analysis, methodology, model performance comparisons, and results**.  

### **Model Implementations**  
Each model has a **separate Jupyter Notebook** for **data preprocessing, training, testing, and tuning**:  
- `i201819-genai-b-a3-q2-vit.ipynb` and `i201819-genai-b-a3-q2-vit-ver-3.ipynb`  → **Vision Transformer (ViT)** implementation for CIFAR-10 classification.  
- `i201819-genai-b-a3-q2-resnet.ipynb` → **ResNet-18 (Transfer Learning)** implementation for CIFAR-10.  
- `i201819-genai-b-a3-q2-cnn-mlp.ipynb` → **CNN-MLP Hybrid model**, where CNN extracts features and MLP classifies them.  

### **Web Application (User Interface & API)**  
- `app.py` → Flask-based **backend API** for **handling model selection, image uploads, and classification requests**.  
- `instructions.html` → Provides **step-by-step guidance** for users on how to select a model, upload images, and view results.  
- `upload.html` → User interface for **selecting a model and uploading multiple images** for classification.  
- `results.html` → Displays **correct and incorrect classifications**, showing **actual images and predicted labels**.  

For now, please refer to **i201819_ImamaAmjad_Ass3.pdf** for **dataset details, model training methodology, and evaluation metrics**. The README will be **expanded later** with additional explanations and improvements.  

---

## Future Enhancements  
- **Expand the README** with dataset preprocessing steps & model architecture explanations
- **Add challenges faced and key learnings** section
- **Improve model performance** by exploring **larger Vision Transformer variants** and **ResNet architectures**
- **Include training results & sample classification outputs**
- **Optimize model training even further with hyperparameter tuning**
- Enhance the web application

---
