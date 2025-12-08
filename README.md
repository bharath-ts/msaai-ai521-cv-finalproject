# Automated Waste Segregation System Using Computer Vision

## Project Overview

This project implements a deep learning image classifier for automated waste segregation using computer vision techniques. The system classifies waste images into 6 categories: cardboard, glass, metal, paper, plastic, and trash. The project demonstrates the effectiveness of transfer learning on a relatively small dataset and provides a complete workflow from data exploration to model deployment.

## Team Information

**Team Number:** Group 3

**Team Members:**
- Jasmeet Kaur
- Himanshu Kumar  
- Bharath TS (Team Leader/Representative)

**GitHub Repository:** [https://github.com/bharath-ts/msaai-ai521-cv-finalproject](https://github.com/bharath-ts/msaai-ai521-cv-finalproject)

## Problem Statement

Manual waste sorting is time-consuming, labor-intensive, and prone to human error. This project addresses the need for automated waste classification systems that can accurately categorize recyclable materials, reducing contamination in recycling streams and improving overall recycling efficiency.

## Dataset

- **Total Images:** 2,527
- **Classes:** 6 (cardboard, glass, metal, paper, plastic, trash)
- **Format:** PNG images
- **Organization:** Class-specific folder structure

## Methodology

### 1. Data Exploration & Analysis
- Class distribution analysis
- Image size and brightness analysis
- Color histogram analysis
- Sample visualization across all classes

### 2. Image Preprocessing
- Image resizing and normalization
- Data augmentation techniques
- Train/validation/test splitting

### 3. Feature Engineering
- Edge detection analysis
- Contour analysis for shape features

### 4. Model Architecture
- **Primary Model:** MobileNetV2 with transfer learning
- **Comparison Models:** ResNet50, EfficientNetB0
- **Class Imbalance Handling:** Class weighting vs. oversampling
- **Training Strategy:** Fine-tuning pre-trained models

### 5. Hyperparameter Optimization
- Learning rate scheduling
- Early stopping configuration
- Extended training experiments

## Results

### Final Model Performance
- **Test Accuracy:** 80.26%
- **Architecture:** MobileNetV2 with simplified classification head
- **Training Strategy:** Extended training with adjusted hyperparameters

### Key Findings
1. **Transfer Learning Effectiveness:** Achieved strong performance with limited data (2,527 images)
2. **Class Imbalance:** Class weighting outperformed oversampling (79.7% vs. 78.4% accuracy)
3. **Architecture Selection:** MobileNetV2 significantly outperformed ResNet50 (79.7% vs. 32.9%) and EfficientNetB0 (79.7% vs. 23.4%)
4. **Model Simplification:** Removing intermediate layers improved performance by reducing overfitting

## Technologies Used

- **Deep Learning:** TensorFlow/Keras
- **Computer Vision:** OpenCV, PIL
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Model Evaluation:** Scikit-learn
- **Environment:** Google Colab with GPU acceleration

## Requirements

pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn pillow## 🚀 Usage

1. **Data Preparation:** Organize images in class-specific folders
2. **Run EDA:** Execute exploratory data analysis cells
3. **Train Model:** Run the training pipeline with your preferred architecture
4. **Evaluate:** Use the evaluation metrics and confusion matrices
5. **Deploy:** Integrate the trained model into your waste sorting application

## Model Performance Insights

### Hyperparameter Learnings
- Early stopping patience and monitoring metrics significantly impact training
- Learning rate reduction strategy affects fine-tuning quality
- Extended training with proper early stopping improves convergence
- Aggressive learning rate reduction (factor=0.2) with longer patience (10 epochs) enables better fine-tuning

### Data Augmentation Impact
- Critical for improving generalization
- Reduces overfitting on smaller datasets
- Enhances model robustness to variations in lighting, orientation, and scale

## Applications

This classifier can be deployed in:
- **Automated waste sorting systems** in recycling facilities
- **Conveyor belt sorting systems** for real-time classification
- **Educational tools** for teaching proper waste classification
- **Mobile applications** for camera-based waste identification
- **Smart bins** for automated waste categorization in public spaces

## Contributing

This is an academic project by Team 3. For questions or collaboration inquiries, please contact the team leader: Bharath TS

**Team Leader:** Bharath TS
**Repository:** [https://github.com/bharath-ts/msaai-ai521-cv-finalproject](https://github.com/bharath-ts/msaai-ai521-cv-finalproject)

---

*Developed as part of MSAI AI521 Computer Vision Course Final Project*
