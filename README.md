# Facial-Emotion-Recognition
Facial Emotion Recognition (FER) is an important computer vision task that enables machines to understand human emotions through facial expressions. This project implements an end-to-end deep learning pipeline using Convolutional Neural Networks (CNNs) to classify facial images into emotional categories. 

## Problem Statement
The goal of this project is to design and evaluate a deep learning–based system that can accurately classify facial expressions into predefined emotion categories. The project emphasizes:

- Proper data preprocessing and augmentation

- Careful CNN architecture design

- Extensive hyperparameter and component analysis

- Robust evaluation across multiple dataset splits

## Emotion Classes
The model classifies facial expressions into the following four emotion categories:
- Angry
- Happy
- Sad
- Neutral

## Dataset Description 
The dataset consists of grayscale facial images, evenly distributed across emotion classes to avoid class imbalance.

**Dataset Splits**
Multiple dataset subsets were used for experimentation and evaluation:

Train_400/     → Hyperparameter tuning

Train_1000/    → Primary training dataset

Test_100/      → Quick validation

Test_250/      → Final evaluation

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed to better understand the dataset characteristics and identify potential challenges before model training. Key analysis steps included:

- Class distribution inspection to verify balanced class representation

- Image visualization to observe visual differences between emotion categories

- Pixel intensity distribution analysis for brightness and contrast patterns

- Noise and variance inspection across classes

## Data Preprocessing
- A structured preprocessing pipeline was designed to prepare the images for training:

- Conversion to 48×48 grayscale format

- Tensor conversion for PyTorch compatibility

- Normalization using dataset mean and standard deviation

- Data augmentation applied only on training samples:

  - Horizontal flipping

  - Random rotations (±10°)

  - Brightness/contrast adjustments

These steps improve robustness and reduce overfitting during training.

## Model Architecture
After experimenting with several CNN configurations, the following architecture was selected as the final model:

- 3 × Convolutional Blocks (with ReLU + MaxPooling + Dropout)

- Fully Connected (512 units + BatchNorm + Dropout)

- Softmax output layer with 4 emotion classes

The final model contained approximately 6 million parameters.

## Experiments and Hyperparameter Tuning
Multiple experiments were conducted to evaluate the impact of:

- Optimizers (Adam, SGD, RMSprop)

- Learning rate schedules

- Batch size variation

- Kernel and pooling configurations

- Dropout and regularization

- Activation functions

## Model Performance
The final model achieved the following results:
| Metric                   | Value   |
| ------------------------ | ------- |
| Best Validation Accuracy | **68%** |
| Training Accuracy        | 79%     |
| Epochs to Convergence    | ~32     |

**Per-Class Accuracy**
| Emotion | Accuracy |
| ------- | -------- |
| Happy   | 92%      |
| Angry   | 68%      |
| Neutral | 64%      |
| Sad     | 40%      |

## Technology Used
- Python

- PyTorch

- NumPy / Pandas

- Matplotlib / Seaborn

- OpenCV

- Jupyter Notebook

## Future Improvements
- Increase emotion categories (e.g., Fear, Disgust, Surprise)

- Use larger and more diverse datasets

- Integrate attention or transformer-based modules

- Deploy as a real-time webcam application

- Improve accuracy for Neutral and Sad classes

## Conclusion
This project demonstrates a complete deep learning pipeline for Facial Emotion Recognition, incorporating preprocessing, CNN-based modeling, experimental evaluation, and analysis. The approach highlights the importance of data quality, architectural choices, and hyperparameter tuning in building effective FER systems.

## Contributors
Anushi Pumbhadiya (github.com/AnushiPumbhadiya08)

Arjun Parikh (github.com/ArjunParikh1404)
