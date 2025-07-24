# 🧪 CNN Hyperparameter Tuning on Fashion MNIST with KerasTuner

This project applies **hyperparameter tuning** to a Convolutional Neural Network (CNN) using [KerasTuner](https://keras.io/keras_tuner/) to classify the Fashion MNIST dataset. Two search strategies are implemented and compared:

- 🎲 **Random Search**
- 🧠 **Bayesian Optimisation**

The goal is to find the best-performing model by systematically tuning hyperparameters such as the number of dense units, dropout rate, learning rate, and optimiser.

---

## 🧠 What is Hyperparameter Tuning?

In machine learning, **hyperparameters** are configuration settings that govern model architecture and training, such as learning rate, batch size, or the number of hidden units. These values aren’t learned during training but must be set manually or discovered through **tuning**.

Manual tuning is time-consuming and error-prone, especially as model complexity increases. Automated hyperparameter tuning helps find better models faster and more consistently.

---

## 🔧 About KerasTuner

[KerasTuner](https://keras.io/keras_tuner/) is a scalable, modular library built for **automated hyperparameter tuning** with Keras models. It provides:

- Built-in tuning strategies like RandomSearch, BayesianOptimization, Hyperband, and Sklearn integration
- Customisable search spaces and objectives
- Clean integration into existing Keras workflows

---

## 🔍 Search Strategies Used

### 🎲 Random Search
Randomly samples hyperparameter combinations from the defined search space. Despite its simplicity, Random Search is surprisingly effective and can outperform grid search when only a few parameters significantly impact performance.

### 🧠 Bayesian Optimisation
Builds a **probabilistic model** of the objective function (e.g., validation loss) and uses it to decide where to sample next. It balances exploration (trying new areas) and exploitation (refining promising areas). It is generally more **sample-efficient** than random search, particularly in high-dimensional spaces.

---

## 📦 Dataset: Fashion MNIST

- 60,000 training and 10,000 test samples
- 28x28 grayscale images
- 10 balanced clothing categories
- Lightweight and ideal for prototyping CNN models

---

## 🛠️ Model & Tuning Details

Both tuning strategies optimise a CNN architecture with the following hyperparameters:

- Number of dense units (e.g., 64–512)
- Dropout rate (0.1–0.5)
- Learning rate (1e-4 to 1e-2)
- Optimiser: Adam, RMSprop
- Number of epochs: 30
- Batch size: 128
- Early stopping used to avoid overfitting

---

## 📊 Results Summary

| Metric / Tuner               | 🎲 Random Search                                  | 🧠 Bayesian Optimisation                      |
|-----------------------------|---------------------------------------------------|-----------------------------------------------|
| **Best Validation Accuracy**| 0.9068 (Trial #4)                                 | 0.9064 (Trial #9)                              |
| **Best Test Accuracy**      | 0.9010                                            | 0.8987                                         |
| **Best Hyperparameters**    | Units: 480, Dropout: 0.3, LR: ~0.000256, Opt: Adam| Units: 224, Dropout: 0.2, LR: ~0.000749, Opt: Adam |
| **Trial with Best Result**  | Trial #4                                          | Trial #9                                       |
| **Total Trials Run**        | 10                                                | 10                                             |
| **Total Elapsed Time**      | 1h 31m 39s                                        | 1h 20m 09s                                     |

---

## 📈 Observations & Analysis

- **Accuracy:** Random Search slightly outperformed Bayesian Optimisation in both test and validation accuracy, although the difference was marginal.
- **Trial Efficiency:** Random Search found its best model early (Trial #4), while Bayesian Optimisation required more iterations (Trial #9).
- **Time:** Bayesian Optimisation finished its 10 trials slightly faster. This may be due to more favorable early stopping or quicker convergence in certain trials.

---

## 🤔 Why Didn’t Bayesian Optimisation Perform Better?

Bayesian Optimisation is generally more efficient, but there are reasons why Random Search held its ground here:

- **Few Trials (10):** Bayesian Optimisation needs enough data to build a reliable surrogate model of the search space. With only 10 trials, its exploratory power is limited.
- **Chance Wins:** Random Search can “get lucky” and land on a good configuration by chance, especially in small search spaces.
- **Search Space Simplicity:** If the hyperparameter relationships are relatively simple or loosely coupled, Random Search can perform competitively.
- **Complex Dependencies:** Bayesian Optimisation excels with non-linear, interdependent hyperparameters — which may not be present in this case.

---

## 📁 Files Included

```bash
cnn-keras-tuning-fashion-mnist/
├── random_search_tuning.py      # RandomSearch tuning with KerasTuner
├── bayesian_tuning.py           # BayesianOptimization tuning with KerasTuner
├── requirements.txt
├── README.md
```


---

## ✅ Key Takeaways

- Both Random Search and Bayesian Optimisation found strong-performing models with ~90% test accuracy.
- In this limited trial setup, Random Search slightly outperformed Bayesian Optimisation.
- KerasTuner is a powerful tool for automating model optimisation and helps streamline experimentation.

---

## 📚 Further Reading

- [KerasTuner Documentation](https://keras.io/keras_tuner/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Practical Bayesian Optimisation](https://arxiv.org/abs/1206.2944)

---

## 📌 License

MIT License
