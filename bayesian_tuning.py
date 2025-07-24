from keras import layers
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras


# --- 1. Load and Preprocess Data ---
print("--- 1. Load and Preprocess Data ---")

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalise pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images to (28, 28, 1) for CNN input
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode labels (for categorical_crossentropy; 10 total classes)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Split training data for validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, 
    y_train, 
    test_size=0.2, 
    random_state=42
)

print(f"x_train shape: {x_train.shape}")    # (48000, 28, 28, 1)
print(f"y_train shape: {y_train.shape}")    # (48000, 10)
print(f"x_val shape: {x_val.shape}")        # (12000, 28, 28, 1)
print(f"y_val shape: {y_val.shape}")        # (12000, 10)
print(f"x_test shape: {x_test.shape}")      # (10000, 28, 28, 1)
print(f"y_test shape: {y_test.shape}")      # (10000, 10)

print("-"*100)

# --- 2. Define the Model-Building Function ---
print("--- 2. Define the Model-Building Function ---")

def build_model(hp):
    # Load the pre-trained MobileNetV2 model
    # Note: We'll freeze the base model here, but you could also tune whether to unfreeze or at what layer to unfreeze.
    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),    # Expected input shape for MobileNetV2
        include_top=False,          # Don't include the classifier head
        weights='imagenet'          # Load weights pre-trained on ImageNet
    )
    base_model.trainable = False    # Keep the pre-trained layers frozen for now

    # Create the functional model
    inputs = keras.Input(shape=(28, 28, 1))

    # --- Preprocessing for MobileNetV2 ---
    # MobileNetV2 expects 3-channel input
    # Convert 1-channel (grayscale) to 3-channels (simulated RGB) by repeating the channel
    x = layers.Lambda(lambda img: tf.repeat(img, 3, axis=-1))(inputs)   # (None, 28, 28, 3)
    # Resize from (28, 28, 3) to (96, 96, 3)
    x = layers.Resizing(96, 96)(x)                                      # (None, 96, 96, 3)

    x = base_model(x, training=False) # Run through the frozen base model

    # Add a Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(x)

    # Tune the number of units (i.e. neurons) in the dense classification head
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    x = layers.Dense(hp_units, activation='relu')(x)

    # Tune the dropout rate
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    x = layers.Dropout(hp_dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # Tune the learning rate for the optimiser
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    # Tune the optimiser
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])

    if hp_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

print("-"*100)

# --- 3. Instantiate the Tuner (Bayesian Optimization Example) ---
print("--- 3. Instantiate the Tuner (Bayesian Optimization Example) ---")

tuner_bayes = kt.BayesianOptimization(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=10,                                          # Keep as 10 different combinations for comparison
    executions_per_trial=1,
    directory='keras_tuner_results',
    project_name='fashion_mnist_cnn_bayesian_optimization', # New project name
    overwrite=True                                          # Set to True to start a fresh search
)

print("\n--- Bayesian Optimization Search Space Summary ---")
tuner_bayes.search_space_summary()

"""
Search space summary
Default search space size: 4
units (Int)
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 512, 'step': 32, 'sampling': 'linear'}
dropout_rate (Float)
{'default': 0.2, 'conditions': [], 'min_value': 0.2, 'max_value': 0.5, 'step': 0.1, 'sampling': 'linear'}
learning_rate (Float)
{'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.01, 'step': None, 'sampling': 'log'}
optimizer (Choice)
{'default': 'adam', 'conditions': [], 'values': ['adam', 'rmsprop'], 'ordered': False}
"""

print("-"*100)

# --- 4. Run the Hyperparameter Search ---
print("\n--- Starting Bayesian Optimization Search ---")
tuner_bayes.search(
    x_train, 
    y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)

"""
Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
128               |128               |units
0.2               |0.2               |dropout_rate
0.0041879         |0.0041879         |learning_rate
rmsprop           |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 57s 38ms/step - accuracy: 0.7949 - loss: 0.6727 - val_accuracy: 0.8582 - val_loss: 0.4238
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 56s 37ms/step - accuracy: 0.8574 - loss: 0.4509 - val_accuracy: 0.8685 - val_loss: 0.4220
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 57s 38ms/step - accuracy: 0.8699 - loss: 0.4234 - val_accuracy: 0.8745 - val_loss: 0.4305
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 57s 38ms/step - accuracy: 0.8729 - loss: 0.4294 - val_accuracy: 0.8744 - val_loss: 0.4932
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 56s 37ms/step - accuracy: 0.8749 - loss: 0.4195 - val_accuracy: 0.8777 - val_loss: 0.4553

Trial 1 Complete [00h 04m 43s]
val_accuracy: 0.8777499794960022

Best val_accuracy So Far: 0.8777499794960022
Total elapsed time: 00h 04m 43s

Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
480               |128               |units
0.2               |0.2               |dropout_rate
0.0019803         |0.0041879         |learning_rate
rmsprop           |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 68s 45ms/step - accuracy: 0.8099 - loss: 0.6346 - val_accuracy: 0.8743 - val_loss: 0.3655
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 77s 52ms/step - accuracy: 0.8771 - loss: 0.3684 - val_accuracy: 0.8857 - val_loss: 0.3446
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 82s 54ms/step - accuracy: 0.8865 - loss: 0.3447 - val_accuracy: 0.8845 - val_loss: 0.3733
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 84s 56ms/step - accuracy: 0.8941 - loss: 0.3280 - val_accuracy: 0.8875 - val_loss: 0.3927
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 89s 59ms/step - accuracy: 0.8951 - loss: 0.3350 - val_accuracy: 0.8870 - val_loss: 0.3899

Trial 2 Complete [00h 06m 40s]
val_accuracy: 0.887499988079071

Best val_accuracy So Far: 0.887499988079071
Total elapsed time: 00h 11m 24s

Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
320               |480               |units
0.2               |0.2               |dropout_rate
0.0064174         |0.0019803         |learning_rate
adam              |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 90s 59ms/step - accuracy: 0.8030 - loss: 0.6159 - val_accuracy: 0.8583 - val_loss: 0.4103
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 91s 61ms/step - accuracy: 0.8535 - loss: 0.4234 - val_accuracy: 0.8679 - val_loss: 0.3896
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 91s 61ms/step - accuracy: 0.8617 - loss: 0.4065 - val_accuracy: 0.8822 - val_loss: 0.3421
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 90s 60ms/step - accuracy: 0.8697 - loss: 0.3762 - val_accuracy: 0.8844 - val_loss: 0.3497
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 87s 58ms/step - accuracy: 0.8745 - loss: 0.3576 - val_accuracy: 0.8802 - val_loss: 0.3743
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 78s 52ms/step - accuracy: 0.8745 - loss: 0.3641 - val_accuracy: 0.8822 - val_loss: 0.3764

Trial 3 Complete [00h 08m 47s]
val_accuracy: 0.8844166398048401

Best val_accuracy So Far: 0.887499988079071
Total elapsed time: 00h 20m 11s

Search: Running Trial #4

Value             |Best Value So Far |Hyperparameter
416               |480               |units
0.4               |0.2               |dropout_rate
0.00016527        |0.0019803         |learning_rate
rmsprop           |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 72s 48ms/step - accuracy: 0.7855 - loss: 0.6274 - val_accuracy: 0.8792 - val_loss: 0.3347
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 74s 49ms/step - accuracy: 0.8801 - loss: 0.3353 - val_accuracy: 0.8917 - val_loss: 0.3040
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 75s 50ms/step - accuracy: 0.8945 - loss: 0.2934 - val_accuracy: 0.8936 - val_loss: 0.2942
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 73s 49ms/step - accuracy: 0.9033 - loss: 0.2739 - val_accuracy: 0.8977 - val_loss: 0.2848
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 74s 49ms/step - accuracy: 0.9063 - loss: 0.2612 - val_accuracy: 0.9006 - val_loss: 0.2858
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 68s 45ms/step - accuracy: 0.9123 - loss: 0.2493 - val_accuracy: 0.9005 - val_loss: 0.2837
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 65s 44ms/step - accuracy: 0.9163 - loss: 0.2383 - val_accuracy: 0.9021 - val_loss: 0.2806
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 65s 44ms/step - accuracy: 0.9193 - loss: 0.2267 - val_accuracy: 0.9037 - val_loss: 0.2869
Epoch 9/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 43ms/step - accuracy: 0.9225 - loss: 0.2181 - val_accuracy: 0.9053 - val_loss: 0.2849
Epoch 10/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 65s 43ms/step - accuracy: 0.9260 - loss: 0.2155 - val_accuracy: 0.9049 - val_loss: 0.2845

Trial 4 Complete [00h 11m 36s]
val_accuracy: 0.9052500128746033

Best val_accuracy So Far: 0.9052500128746033
Total elapsed time: 00h 31m 47s

Search: Running Trial #5

Value             |Best Value So Far |Hyperparameter
32                |416               |units
0.2               |0.4               |dropout_rate
0.00016415        |0.00016527        |learning_rate
rmsprop           |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 41ms/step - accuracy: 0.6700 - loss: 0.9566 - val_accuracy: 0.8602 - val_loss: 0.3902
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.8567 - loss: 0.4271 - val_accuracy: 0.8727 - val_loss: 0.3511
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 61s 41ms/step - accuracy: 0.8673 - loss: 0.3784 - val_accuracy: 0.8812 - val_loss: 0.3318
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 62s 41ms/step - accuracy: 0.8823 - loss: 0.3403 - val_accuracy: 0.8869 - val_loss: 0.3149
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 61s 41ms/step - accuracy: 0.8883 - loss: 0.3251 - val_accuracy: 0.8891 - val_loss: 0.3137
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 60s 40ms/step - accuracy: 0.8918 - loss: 0.3150 - val_accuracy: 0.8898 - val_loss: 0.3083
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 60s 40ms/step - accuracy: 0.8972 - loss: 0.2988 - val_accuracy: 0.8913 - val_loss: 0.3056
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 60s 40ms/step - accuracy: 0.9001 - loss: 0.2949 - val_accuracy: 0.8945 - val_loss: 0.2992
Epoch 9/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 61s 41ms/step - accuracy: 0.9006 - loss: 0.2887 - val_accuracy: 0.8940 - val_loss: 0.2997
Epoch 10/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 59s 39ms/step - accuracy: 0.8993 - loss: 0.2902 - val_accuracy: 0.8956 - val_loss: 0.3001

Trial 5 Complete [00h 10m 11s]
val_accuracy: 0.8955833315849304

Best val_accuracy So Far: 0.9052500128746033
Total elapsed time: 00h 41m 58s

Search: Running Trial #6

Value             |Best Value So Far |Hyperparameter
192               |416               |units
0.2               |0.4               |dropout_rate
0.0012017         |0.00016527        |learning_rate
adam              |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 60s 40ms/step - accuracy: 0.8179 - loss: 0.5084 - val_accuracy: 0.8760 - val_loss: 0.3272
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 60s 40ms/step - accuracy: 0.8864 - loss: 0.3072 - val_accuracy: 0.8847 - val_loss: 0.3158
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 60s 40ms/step - accuracy: 0.8952 - loss: 0.2759 - val_accuracy: 0.8934 - val_loss: 0.2943
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 59s 40ms/step - accuracy: 0.9053 - loss: 0.2532 - val_accuracy: 0.8940 - val_loss: 0.2803
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 62s 41ms/step - accuracy: 0.9080 - loss: 0.2456 - val_accuracy: 0.8950 - val_loss: 0.2823
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 62s 41ms/step - accuracy: 0.9153 - loss: 0.2227 - val_accuracy: 0.8987 - val_loss: 0.2816
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 61s 40ms/step - accuracy: 0.9212 - loss: 0.2101 - val_accuracy: 0.8932 - val_loss: 0.2929

Trial 6 Complete [00h 07m 04s]
val_accuracy: 0.8986666798591614

Best val_accuracy So Far: 0.9052500128746033
Total elapsed time: 00h 49m 02s

Search: Running Trial #7

Value             |Best Value So Far |Hyperparameter
480               |416               |units
0.3               |0.4               |dropout_rate
0.00010384        |0.00016527        |learning_rate
rmsprop           |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 66s 43ms/step - accuracy: 0.7713 - loss: 0.6527 - val_accuracy: 0.8744 - val_loss: 0.3442
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.8838 - loss: 0.3258 - val_accuracy: 0.8838 - val_loss: 0.3154
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 43ms/step - accuracy: 0.8938 - loss: 0.2953 - val_accuracy: 0.8923 - val_loss: 0.2899
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 43ms/step - accuracy: 0.9039 - loss: 0.2694 - val_accuracy: 0.8958 - val_loss: 0.2832
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.9111 - loss: 0.2453 - val_accuracy: 0.8992 - val_loss: 0.2753
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 65s 43ms/step - accuracy: 0.9156 - loss: 0.2366 - val_accuracy: 0.9013 - val_loss: 0.2760
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 43ms/step - accuracy: 0.9210 - loss: 0.2250 - val_accuracy: 0.9013 - val_loss: 0.2734
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 42ms/step - accuracy: 0.9236 - loss: 0.2116 - val_accuracy: 0.9017 - val_loss: 0.2787
Epoch 9/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 43ms/step - accuracy: 0.9265 - loss: 0.2065 - val_accuracy: 0.9041 - val_loss: 0.2725
Epoch 10/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 43ms/step - accuracy: 0.9297 - loss: 0.1982 - val_accuracy: 0.9053 - val_loss: 0.2753

Trial 7 Complete [00h 10m 41s]
val_accuracy: 0.9052500128746033

Best val_accuracy So Far: 0.9052500128746033
Total elapsed time: 00h 59m 43s

Search: Running Trial #8

Value             |Best Value So Far |Hyperparameter
192               |416               |units
0.3               |0.4               |dropout_rate
0.0038562         |0.00016527        |learning_rate
rmsprop           |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.7806 - loss: 0.7819 - val_accuracy: 0.8641 - val_loss: 0.4085
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 42ms/step - accuracy: 0.8530 - loss: 0.4658 - val_accuracy: 0.8685 - val_loss: 0.4301
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 43ms/step - accuracy: 0.8612 - loss: 0.4612 - val_accuracy: 0.8723 - val_loss: 0.4660
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.8641 - loss: 0.4726 - val_accuracy: 0.8708 - val_loss: 0.4621

Trial 8 Complete [00h 04m 15s]
val_accuracy: 0.8723333477973938

Best val_accuracy So Far: 0.9052500128746033
Total elapsed time: 01h 03m 58s

Search: Running Trial #9

Value             |Best Value So Far |Hyperparameter
224               |416               |units
0.2               |0.4               |dropout_rate
0.0007494         |0.00016527        |learning_rate
adam              |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 65s 43ms/step - accuracy: 0.8218 - loss: 0.5060 - val_accuracy: 0.8726 - val_loss: 0.3416
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.8900 - loss: 0.3010 - val_accuracy: 0.8857 - val_loss: 0.3042
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.9020 - loss: 0.2642 - val_accuracy: 0.8849 - val_loss: 0.3209
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.9114 - loss: 0.2365 - val_accuracy: 0.8968 - val_loss: 0.2796
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 42ms/step - accuracy: 0.9173 - loss: 0.2205 - val_accuracy: 0.8927 - val_loss: 0.2912
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 42ms/step - accuracy: 0.9265 - loss: 0.1992 - val_accuracy: 0.9017 - val_loss: 0.2760
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 64s 42ms/step - accuracy: 0.9307 - loss: 0.1848 - val_accuracy: 0.9027 - val_loss: 0.2791
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.9311 - loss: 0.1802 - val_accuracy: 0.8984 - val_loss: 0.2870
Epoch 9/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 63s 42ms/step - accuracy: 0.9375 - loss: 0.1626 - val_accuracy: 0.9064 - val_loss: 0.2950

Trial 9 Complete [00h 09m 33s]
val_accuracy: 0.906416654586792

Best val_accuracy So Far: 0.906416654586792
Total elapsed time: 01h 13m 31s

Search: Running Trial #10

Value             |Best Value So Far |Hyperparameter
512               |224               |units
0.3               |0.2               |dropout_rate
0.0013763         |0.0007494         |learning_rate
rmsprop           |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 67s 44ms/step - accuracy: 0.8107 - loss: 0.5859 - val_accuracy: 0.8735 - val_loss: 0.3602
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 66s 44ms/step - accuracy: 0.8763 - loss: 0.3645 - val_accuracy: 0.8831 - val_loss: 0.3543
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 66s 44ms/step - accuracy: 0.8882 - loss: 0.3416 - val_accuracy: 0.8912 - val_loss: 0.3333
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 67s 44ms/step - accuracy: 0.8913 - loss: 0.3327 - val_accuracy: 0.8881 - val_loss: 0.3894
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 66s 44ms/step - accuracy: 0.8975 - loss: 0.3193 - val_accuracy: 0.8959 - val_loss: 0.3416
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 67s 45ms/step - accuracy: 0.9044 - loss: 0.3163 - val_accuracy: 0.8909 - val_loss: 0.3797

Trial 10 Complete [00h 06m 38s]
val_accuracy: 0.8959166407585144

Best val_accuracy So Far: 0.906416654586792
Total elapsed time: 01h 20m 09s
"""

print("-"*100)

# --- 5. Retrieve the Best Model and Hyperparameters ---
print("\n--- Bayesian Optimization Results ---")
best_hps_bayes = tuner_bayes.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters found by Bayesian Optimization:")
print(f"  Units in Dense layer: {best_hps_bayes.get('units')}")     # 224
print(f"  Dropout Rate: {best_hps_bayes.get('dropout_rate')}")      # 0.2
print(f"  Learning Rate: {best_hps_bayes.get('learning_rate')}")    # 0.0007493975222254006
print(f"  Optimizer: {best_hps_bayes.get('optimizer')}")            # adam

best_model_bayes = tuner_bayes.get_best_models(num_models=1)[0]
print(f"Best model's summary (from Bayesian Optimization):")
best_model_bayes.summary()

"""
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)           │ (None, 28, 28, 1)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lambda (Lambda)                      │ (None, 28, 28, 3)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ resizing (Resizing)                  │ (None, 96, 96, 3)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ mobilenetv2_1.00_96 (Functional)     │ (None, 3, 3, 1280)          │       2,257,984 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 1280)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 224)                 │         286,944 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 224)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           2,250 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,547,178 (9.72 MB)
 Trainable params: 289,194 (1.10 MB)
 Non-trainable params: 2,257,984 (8.61 MB)
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 37ms/step - accuracy: 0.8980 - loss: 0.3250  
"""

# Evaluate the best model on test data
loss_bayes, accuracy_bayes = best_model_bayes.evaluate(x_test, y_test)
print(f"Best model from Bayesian Optimisation - Test Loss: {loss_bayes:.4f}, Test Accuracy: {accuracy_bayes:.4f}")
# Best model from Bayesian Optimisation - Test Loss: 0.3182, Test Accuracy: 0.8987

"""
Summary:

Metric / Tuner	                |Random Search	                                     |Bayesian Optimisation
--------------------------------|----------------------------------------------------|-----------------------
Best Validation Accuracy      | 0.9068 (Trial #4)	                                   | 0.9064 (Trial #9)
Best Test Accuracy	          | 0.9010	                                             | 0.8987
Best Hyperparameters          | Units: 480, Dropout: 0.3, LR: ~0.000256, Opt: Adam   | Units: 224, Dropout: 0.2, LR: ~0.000749, Opt: Adam
Trial where best was found    | Trial #4	                                           | Trial #9
Total Trials Run	            | 10	                                                 | 10
Total Elapsed Time	          | 1h 31m 39s	                                         | 1h 20m 09s

Observations:
- Accuracy: Random Search achieved a slightly higher test accuracy (0.9010 vs. 0.8987) and validation accuracy (0.9068 vs. 0.9064)
- Best Trial: Interestingly, Bayesian Optimisation found its best result on Trial #9, while Random Search found its best earlier in Trial #4.
- Time: Bayesian Optimisation completed its 10 trials a bit faster (1h 20m vs. 1h 31m). This could be due to the specific hyperparameter combinations it chose leading to slightly quicker convergence or early stopping.

While Bayesian Optimisation is theoretically more efficient and often finds better results with fewer trials on average, it's not guaranteed to always outperform Random Search, especially when:
- Small Number of Trials: With only 10 trials, the "smart" search of Bayesian Optimisation might not have enough data to build a truly accurate surrogate model of the search space. 
    Random Search can sometimes "get lucky" and hit a good combination by chance within a limited number of tries.
- Search Space Characteristics: For some search spaces, a truly random exploration might stumble upon optima that the surrogate model of Bayesian Optimisation initially dismisses or doesn't explore sufficiently early on.
- Hyperparameter Dependencies: Bayesian Optimisation excels when there are complex, non-linear dependencies between hyperparameters. 
    If the relationship is simpler or more direct, Random Search can still be very effective.

"""
