====================================================================
TASK 2: FACIAL EMOTION RECOGNITION (FER)
====================================================================

1. PROBLEM INTRODUCTION AND SOLUTION
- Problem: Facial Emotion Recognition (FER) from images. The system classifies images into 6 main emotion categories: Angry, Fear, Happy, Neutral, Sad, Surprise (the Disgust label was removed due to an extremely small number of samples, causing severe data imbalance).
- Reason for choosing the solution: Deep Learning (specifically Convolutional Neural Networks - CNNs) is currently the most powerful method for static image processing tasks.
- Method:
  Using Transfer Learning with 3 popular CNN architectures pre-trained on ImageNet:
    + EfficientNet-B2: The main model selected because it achieves the best balance between accuracy (SOTA) and the number of parameters (Params: ~9.1M).
    + ResNet-50: The baseline model, featuring a deep and popular architecture for comparison (Params: ~24M).
    + MobileNetV3-Large: The most lightweight model, optimized for real-time applications (Params: ~5.4M).
  The training process (Fine-tuning) is refined through 2 phases:
    + Phase 1 (Warm-up): Freeze the feature extraction layers (backbone), only train the new classification layer (Classification Head).
    + Phase 2 (Fine-tune): Unfreeze all (or part of) the layers to train with a low learning rate.

2. DATASET
- Using the FER-2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013
- Input image size: Resized to 224x224 (RGB) with ImageNet normalization standards.
- Dataset structure:
    + Training set: ~28,273 images. Using WeightedRandomSampler and strong Augmentation techniques to solve the class imbalance problem.
    + Test/Holdout set: ~7,067 images.

3. DIRECTORY STRUCTURE
The project includes the following main files and directories:
├── task2.ipynb             # Kaggle Notebook (Python) containing all source code for data processing (EDA, Dataloader), model building, Training, and Evaluation.
├── config.json             # JSON file storing configurations (label names, normalization parameters) and accuracy evaluation (Accuracy, F1) of the 3 models.
├── index.html              # Web Interface (Front-end) Demo for recognition.
├── efficientnet_b2.pth     # Trained weight checkpoint for the EfficientNet-B2 model.
├── resnet50.pth            # Trained weight checkpoint for the ResNet-50 model.
├── mobilenet_v3.pth        # Trained weight checkpoint for the MobileNetV3 model.
└── README.txt              # This instruction file.

4. INSTALLATION AND USAGE INSTRUCTIONS
a. Training environment:
- Platform: Highly recommend using Kaggle Notebook with GPU (Tesla T4) to optimize runtime.
- Required libraries: torch, torchvision, torchinfo, numpy, pandas, matplotlib, seaborn, scikit-learn, cv2. (Use the command `!pip install torchinfo` if not already installed).

b. Running the training source code:
- Upload the FER-2013 dataset folder to the Kaggle environment at the path: `/kaggle/input/msambare/fer2013`.
- Open the `task2.ipynb` file and execute "Run All". The notebook will sequentially load data, build the models, and train them.
- The training history and weight checkpoints (`.pth`) will be exported as a ZIP file (`fer_demo_models.zip`) into the outputs folder.

c. Running the Demo interface:
- After obtaining the weight files (.pth) and `config.json`, place them in the same directory as the `index.html` file.
- Open the `index.html` file directly using a web browser to upload an image and see the model's prediction of facial expressions.

5. EVALUATION RESULTS
Based on empirical experiments:
- EfficientNet-B2: Yields the most comprehensive results (Accuracy: ~59.2%, Macro F1: ~0.57).
- ResNet-50: Consumes the most parameters but performs similarly or slightly worse as it is prone to overfitting.
- MobileNetV3-Large: Achieves comparable accuracy (Accuracy: ~59.1%) while offering highly optimized inference speed and parameter volume for non-GPU environments.
