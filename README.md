# Violence Detection in Video Surveillance

## Introduction

Public safety is a critical concern, especially in densely populated or high-risk environments such as shopping malls, public transport stations, and urban streets. Traditional surveillance systems rely heavily on manual monitoring, which is time-consuming, prone to human error, and inefficient in detecting violent incidents promptly. 

This project proposes an automated violence detection system using deep learning techniques to analyze video footage and detect aggressive behavior. The system aims to enhance surveillance by providing real-time alerts for potentially violent events, enabling swift and effective responses.

The Real-Life Violence Dataset, consisting of 2,000 videos evenly split between Violence and NonViolence classes, was used. These videos represent real-world scenarios including fighting, sports, daily activities, and crowd interactions.

Three deep learning architectures were implemented and compared:

- **MobileNet + BiLSTM:** A lightweight, efficient combination of a pre-trained CNN and a bidirectional LSTM for temporal pattern learning.
- **VGG16 + LSTM:** A classical CNN architecture for spatial feature extraction followed by LSTM for sequence modeling.
- **Vision Transformer (ViT):** A transformer-based model capturing both spatial and temporal relationships using attention mechanisms.

## Related Work

Automated violence detection has advanced significantly with deep learning. Early methods used handcrafted features and shallow classifiers but lacked robustness. CNNs improved spatial feature extraction, and RNNs, especially LSTMs, enabled temporal modeling.

Hybrid CNN-LSTM models, such as VGG16 + LSTM and MobileNet + BiLSTM, balance accuracy and computational efficiency. Recently, transformer-based models like ViT have emerged, excelling in modeling long-range dependencies in video data.

Datasets like Hockey Fight, Movies Fight, RWF-2000, and the Real-Life Violence Dataset aid development and benchmarking. Challenges remain in computational cost and real-time scalability, addressed here by comparing three architectures on the Real-Life Violence Dataset.

## Data and Methods

### Dataset

- **Total Videos:** 2,000  
- **Violence Class:** 1,000 videos with violent activities  
- **NonViolence Class:** 1,000 videos with peaceful activities  
- **Format:** AVI  
- **Frame Extraction:** 16 equally spaced frames per video

### Preprocessing

- Frames resized to 64×64 pixels  
- Pixel values normalized to [0,1]  
- Labels one-hot encoded  
- Dataset split into training, validation, and test sets with shuffling

### Model Architectures

#### VGG16 + LSTM

- Pre-trained VGG16 as feature extractor (frozen layers, TimeDistributed)  
- LSTM with 64 units for temporal modeling  
- Dense layers with dropout for classification  
- Optimizer: SGD  
- Total parameters: 15,272,770 (trainable: 558,082)

#### MobileNet + BiLSTM

- Pre-trained MobileNet (TimeDistributed) as feature extractor  
- Bidirectional LSTM with 64 units  
- Multiple Dense + Dropout layers  
- Optimizer: SGD  
- Total parameters: 3,637,090 (trainable: 3,060,642)

#### Vision Transformer (ViT + LSTM)

- ResNet50 pre-trained backbone as feature extractor (TimeDistributed)  
- LSTM with 128 units  
- Dense classification layers  
- Optimizer: Adam (learning rate 1e-4)  
- Total parameters: 24,710,722 (trainable: 24,657,602)

### Training and Evaluation

- All models trained for 10 epochs, batch size 8, with 20% validation split  
- Loss: categorical cross-entropy (MobileNet, VGG16), sparse categorical cross-entropy (ViT)  
- Callbacks: EarlyStopping and ReduceLROnPlateau  
- Training hardware: GPU  
- Average training time per epoch: MobileNet + BiLSTM (~150s), VGG16 + LSTM (~450s), ViT (~1300s)

## Results

| Model                    | Test Accuracy | Test Loss |
|--------------------------|---------------|-----------|
| MobileNet + BiLSTM       | 89.45%        | 0.2138    |
| VGG16 + LSTM             | 70.00%        | 0.5864    |
| Vision Transformer + LSTM | 92.75%        | 0.1506    |

The ViT-based model achieved the highest accuracy, followed closely by MobileNet + BiLSTM. The VGG16-based model performed lower, likely due to architectural limitations on low-resolution frames.

## Conclusion

This project demonstrates effective real-time violence detection in videos using deep learning. The Vision Transformer architecture (ViT + LSTM) showed superior performance, followed by MobileNet + BiLSTM. The classical VGG16 + LSTM model underperformed in this context.

Transformer-based models, even with moderate training and resolution, are highly effective for violence detection tasks. These findings support the use of advanced spatial-temporal deep learning models for enhancing public safety surveillance systems.

---
