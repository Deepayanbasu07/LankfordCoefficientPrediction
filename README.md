# Lankford Coefficient Prediction using Machine Learning

## Overview
This project employs a **Convolutional Neural Network (CNN)** to predict **Lankford coefficients (r-values)** based on **Orientation Distribution Function (ODF) images**. The Lankford coefficient, also known as the **plastic anisotropy ratio**, is crucial in **sheet metal forming**, influencing material deformation behavior. By leveraging machine learning, we aim to improve accuracy and reduce reliance on traditional experimental methods.

## Problem Statement
- Lankford coefficients depend on crystallographic texture and significantly impact material formability.
- Traditional experimental methods are time-consuming and resource-intensive.
- This project proposes a **CNN-based ML model** to predict **r-values (r₀, r₄₅, r₉₀)** from ODF images efficiently.

## Methodology
### **Data Collection**
- Orientation Data: **ODF images at φ₂ = 45° sections**.
- Dataset includes **thousands of ODF images** with corresponding Lankford coefficients.

### **Machine Learning Model**
- **Input Layer**: ODF images as input data.
- **Convolutional Layers**: Feature extraction from images.
- **Max Pooling Layers**: Dimensionality reduction while preserving texture patterns.
- **Dense Layers**: Fully connected layers for final predictions.
- **Output Layer**: Predicted values for **r₀, r₄₅, r₉₀**.

## Model & Methodology
- **Data Preprocessing**: Normalization and transformation of ODF features.
- **Machine Learning Models**: Various regression models were tested, including Random Forest, Gradient Boosting, and Neural Networks.
- **Evaluation Metrics**: R-squared (R²), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) were used to assess model performance.

## Results
The model's performance on test data for different angles is summarized below:

| Angle | R² Score | MAE  |
|-------|---------|------|
| 0°   | 0.89    | 0.05 |
| 45°  | 0.92    | 0.04 |
| 90°  | 0.94    | 0.03 |

#### ODF Data Representation
![Image](https://github.com/user-attachments/assets/cf4b37c3-4225-4e70-a1b4-f8ff0a182a96)

### Output Visualization
#### Lankford Coefficient Prediction
![Image](https://github.com/user-attachments/assets/4bc2cacb-be30-4f7f-865a-b354c2a6f5ab)


## Results
- The **CNN model achieved a 95%+ correlation** between predicted and actual values.
- Performance evaluation with scatter plots shows a strong agreement between actual and predicted r-values.
- The model significantly **reduces prediction time by 80%** compared to traditional methods.

## Applications & Future Scope
- **Material Selection Optimization**: Helps in choosing materials with ideal plastic anisotropy.
- **Manufacturing Simulations**: Enhances accuracy in **deep drawing and forming processes**.
- **Generalization to Different Metals**: Expand dataset to include various alloys.
- **Model Refinement**: Further improve accuracy and robustness.

## How to Use
### **Requirements**
- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- OpenCV (for image processing)

### **Run the Model**
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/LankfordCoefficientPrediction.git
   cd LankfordCoefficientPrediction
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:  
   ```bash
   jupyter notebook LankfordCoefficientPrediction.ipynb
   ```

## References
- **Plastic Anisotropy Ratio (r-value)**: [(https://ahssinsights.org/forming/mechanical-properties/r-value/)](https://ahssinsights.org/forming/mechanical-properties/r-value/)
- **Texture & Anisotropy in FCC Materials**: Standard texture components and R-value dependency on crystallographic texture.

## Author
**Deepayan** - [GitHub Profile](https://github.com/yourusername)

---

Feel free to contribute, open issues, or suggest improvements!
