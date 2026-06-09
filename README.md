# Perovskite Bandgap Prediction Model

An end-to-end Machine Learning project that predicts the electronic bandgap of perovskite materials using a K-Nearest Neighbors (KNN) regression framework. 

## 📌 Project Overview
Predicting the bandgap of perovskites is crucial for discovering high-efficiency materials for solar cells, LEDs, and other optoelectronic devices. Traditional computational methods (like DFT) are resource-intensive. This project demonstrates how a data-driven Machine Learning approach can instantly estimate bandgap values based on material properties, speeding up the materials discovery pipeline.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.13
* **Machine Learning:** Scikit-learn (KNN Regressor)
* **Data Handling:** Pandas, NumPy
* **Model Deployment/Persistence:** Pickle (`model.pkl`)

---

## 📂 Repository Structure
```text
├── perovskite_bandgap_60000rows (1).csv  # Dataset containing material features and target bandgaps
├── train_model.py                        # Script to load data, preprocess, and train the KNN model
├── predict.py                            # Inference script to make predictions on new data
├── app.py                                # Main application entry point 
├── utils.py                              # Helper functions for data cleaning and transformations
├── model.pkl                             # Trained and serialized KNN model object (Git LFS tracked)
└── README.md                             # Project documentation
