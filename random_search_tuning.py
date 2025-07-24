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

# --- 3. Instantiate the Tuner (Random Search Example) ---
print("--- 3. Instantiate the Tuner (Random Search Example) ---")

# Define the tuner.
# objective: What metric to optimise (val_accuracy for validation accuracy)
# max_trials: The total number of hyperparameter combinations to test.
# executions_per_trial: How many times to train each model configuration.
#                       (Useful to reduce variance if training is noisy).
#                       Set to 1 for faster tuning.
# directory: Where to save the tuning results (logs, best models).
# project_name: Name for the specific tuning project within the directory.

tuner_random = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=10,          # Test 10 different combinations randomly
    executions_per_trial=1, # Each combination trained once
    directory='keras_tuner_results',
    project_name='fashion_mnist_cnn_random_search',
    overwrite=True          # Set to True to start a fresh search, False to resume
)

print("\n--- Random Search Space Summary ---")
tuner_random.search_space_summary()

# Search space summary
# Default search space size: 4
# units (Int)
# {'default': None, 'conditions': [], 'min_value': 32, 'max_value': 512, 'step': 32, 'sampling': 'linear'}
# dropout_rate (Float)
# {'default': 0.2, 'conditions': [], 'min_value': 0.2, 'max_value': 0.5, 'step': 0.1, 'sampling': 'linear'}
# learning_rate (Float)
# {'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.01, 'step': None, 'sampling': 'log'}
# optimizer (Choice)
# {'default': 'adam', 'conditions': [], 'values': ['adam', 'rmsprop'], 'ordered': False}

print("-"*100)

# --- 4. Run the Hyperparameter Search ---
print("--- 4. Run the Hyperparameter Search ---")

print("\n--- Starting Random Search ---")
# The arguments passed to tuner.search() are passed directly to model.fit()
tuner_random.search(
    x_train, 
    y_train,
    epochs=10,                                                                      
    validation_data=(x_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]    # Stop early if no improvement
)

"""
--- Starting Random Search ---

Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
192               |192               |units
0.2               |0.2               |dropout_rate
0.0044279         |0.0044279         |learning_rate
rmsprop           |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 58s 38ms/step - accuracy: 0.7899 - loss: 0.7754 - val_accuracy: 0.8681 - val_loss: 0.3936
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 56s 37ms/step - accuracy: 0.8599 - loss: 0.4476 - val_accuracy: 0.8747 - val_loss: 0.4223
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 58s 39ms/step - accuracy: 0.8669 - loss: 0.4343 - val_accuracy: 0.8756 - val_loss: 0.4685
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 58s 39ms/step - accuracy: 0.8724 - loss: 0.4357 - val_accuracy: 0.8770 - val_loss: 0.4952

Trial 1 Complete [00h 03m 51s]
val_accuracy: 0.8769999742507935

Best val_accuracy So Far: 0.8769999742507935
Total elapsed time: 00h 03m 51s

Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
288               |192               |units
0.4               |0.2               |dropout_rate
0.0014712         |0.0044279         |learning_rate
adam              |rmsprop           |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 59s 38ms/step - accuracy: 0.8085 - loss: 0.5551 - val_accuracy: 0.8700 - val_loss: 0.3480
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 57s 38ms/step - accuracy: 0.8712 - loss: 0.3513 - val_accuracy: 0.8813 - val_loss: 0.3234
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 58s 38ms/step - accuracy: 0.8834 - loss: 0.3238 - val_accuracy: 0.8890 - val_loss: 0.3011
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 56s 38ms/step - accuracy: 0.8899 - loss: 0.2979 - val_accuracy: 0.8931 - val_loss: 0.2917
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 58s 38ms/step - accuracy: 0.8946 - loss: 0.2834 - val_accuracy: 0.8959 - val_loss: 0.2861
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 57s 38ms/step - accuracy: 0.8966 - loss: 0.2770 - val_accuracy: 0.8929 - val_loss: 0.2936
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 58s 39ms/step - accuracy: 0.9007 - loss: 0.2644 - val_accuracy: 0.8957 - val_loss: 0.2939
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 58s 38ms/step - accuracy: 0.9033 - loss: 0.2540 - val_accuracy: 0.9008 - val_loss: 0.2812
Epoch 9/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 57s 38ms/step - accuracy: 0.9079 - loss: 0.2468 - val_accuracy: 0.9028 - val_loss: 0.2818
Epoch 10/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 59s 39ms/step - accuracy: 0.9096 - loss: 0.2419 - val_accuracy: 0.8952 - val_loss: 0.3123

Trial 2 Complete [00h 09m 36s]
val_accuracy: 0.9028333425521851

Best val_accuracy So Far: 0.9028333425521851
Total elapsed time: 00h 13m 26s

Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
160               |288               |units
0.2               |0.4               |dropout_rate
0.00023127        |0.0014712         |learning_rate
rmsprop           |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 62s 41ms/step - accuracy: 0.8061 - loss: 0.5662 - val_accuracy: 0.8776 - val_loss: 0.3336
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 60s 40ms/step - accuracy: 0.8856 - loss: 0.3189 - val_accuracy: 0.8822 - val_loss: 0.3212
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 62s 41ms/step - accuracy: 0.8962 - loss: 0.2904 - val_accuracy: 0.8914 - val_loss: 0.3038
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 62s 41ms/step - accuracy: 0.9064 - loss: 0.2629 - val_accuracy: 0.8969 - val_loss: 0.2873
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 61s 41ms/step - accuracy: 0.9147 - loss: 0.2456 - val_accuracy: 0.8968 - val_loss: 0.2924
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 62s 41ms/step - accuracy: 0.9192 - loss: 0.2299 - val_accuracy: 0.9020 - val_loss: 0.2880
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 65s 43ms/step - accuracy: 0.9197 - loss: 0.2230 - val_accuracy: 0.9022 - val_loss: 0.2921

Trial 3 Complete [00h 07m 14s]
val_accuracy: 0.9022499918937683

Best val_accuracy So Far: 0.9028333425521851
Total elapsed time: 00h 20m 41s

Search: Running Trial #4

Value             |Best Value So Far |Hyperparameter
480               |288               |units
0.3               |0.4               |dropout_rate
0.00025615        |0.0014712         |learning_rate
adam              |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 76s 50ms/step - accuracy: 0.8040 - loss: 0.5590 - val_accuracy: 0.8831 - val_loss: 0.3177
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 82s 55ms/step - accuracy: 0.8901 - loss: 0.2969 - val_accuracy: 0.8903 - val_loss: 0.2927
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 88s 59ms/step - accuracy: 0.9063 - loss: 0.2593 - val_accuracy: 0.8992 - val_loss: 0.2758
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 88s 59ms/step - accuracy: 0.9139 - loss: 0.2350 - val_accuracy: 0.8978 - val_loss: 0.2796
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 92s 61ms/step - accuracy: 0.9189 - loss: 0.2143 - val_accuracy: 0.9068 - val_loss: 0.2589
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 99s 66ms/step - accuracy: 0.9262 - loss: 0.1982 - val_accuracy: 0.9032 - val_loss: 0.2650
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 98s 66ms/step - accuracy: 0.9314 - loss: 0.1820 - val_accuracy: 0.9049 - val_loss: 0.2617
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 104s 69ms/step - accuracy: 0.9375 - loss: 0.1671 - val_accuracy: 0.9031 - val_loss: 0.2733

Trial 4 Complete [00h 12m 09s]
val_accuracy: 0.9068333506584167

Best val_accuracy So Far: 0.9068333506584167
Total elapsed time: 00h 32m 49s

Search: Running Trial #5

Value             |Best Value So Far |Hyperparameter
192               |480               |units
0.3               |0.3               |dropout_rate
0.0054616         |0.00025615        |learning_rate
rmsprop           |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 98s 65ms/step - accuracy: 0.7772 - loss: 0.8466 - val_accuracy: 0.8484 - val_loss: 0.4415
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 99s 66ms/step - accuracy: 0.8437 - loss: 0.5327 - val_accuracy: 0.8630 - val_loss: 0.4744
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 99s 66ms/step - accuracy: 0.8488 - loss: 0.5299 - val_accuracy: 0.8769 - val_loss: 0.5146
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 103s 69ms/step - accuracy: 0.8547 - loss: 0.5235 - val_accuracy: 0.8708 - val_loss: 0.4892

Trial 5 Complete [00h 06m 40s]
val_accuracy: 0.8769166469573975

Best val_accuracy So Far: 0.9068333506584167
Total elapsed time: 00h 39m 29s

Search: Running Trial #6

Value             |Best Value So Far |Hyperparameter
64                |480               |units
0.3               |0.3               |dropout_rate
0.007368          |0.00025615        |learning_rate
adam              |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 95s 63ms/step - accuracy: 0.7705 - loss: 0.6717 - val_accuracy: 0.8608 - val_loss: 0.4021
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 90s 60ms/step - accuracy: 0.8316 - loss: 0.4827 - val_accuracy: 0.8618 - val_loss: 0.4034
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 89s 59ms/step - accuracy: 0.8363 - loss: 0.4724 - val_accuracy: 0.8675 - val_loss: 0.4357
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 91s 60ms/step - accuracy: 0.8457 - loss: 0.4562 - val_accuracy: 0.8785 - val_loss: 0.3534
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 89s 60ms/step - accuracy: 0.8457 - loss: 0.4466 - val_accuracy: 0.8712 - val_loss: 0.4015
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 90s 60ms/step - accuracy: 0.8465 - loss: 0.4484 - val_accuracy: 0.8763 - val_loss: 0.3679
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 91s 61ms/step - accuracy: 0.8507 - loss: 0.4288 - val_accuracy: 0.8720 - val_loss: 0.3813

Trial 6 Complete [00h 10m 36s]
val_accuracy: 0.8784999847412109

Best val_accuracy So Far: 0.9068333506584167
Total elapsed time: 00h 50m 05s

Search: Running Trial #7

Value             |Best Value So Far |Hyperparameter
64                |480               |units
0.2               |0.3               |dropout_rate
0.00075489        |0.00025615        |learning_rate
rmsprop           |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 96s 63ms/step - accuracy: 0.8012 - loss: 0.5684 - val_accuracy: 0.8734 - val_loss: 0.3492
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 93s 62ms/step - accuracy: 0.8836 - loss: 0.3297 - val_accuracy: 0.8874 - val_loss: 0.3276
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 94s 62ms/step - accuracy: 0.8947 - loss: 0.3051 - val_accuracy: 0.8910 - val_loss: 0.3276
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 92s 62ms/step - accuracy: 0.8982 - loss: 0.2919 - val_accuracy: 0.8942 - val_loss: 0.3249
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 93s 62ms/step - accuracy: 0.9024 - loss: 0.2826 - val_accuracy: 0.8940 - val_loss: 0.3514
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 91s 61ms/step - accuracy: 0.9071 - loss: 0.2812 - val_accuracy: 0.8922 - val_loss: 0.3517
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 92s 61ms/step - accuracy: 0.9104 - loss: 0.2727 - val_accuracy: 0.8935 - val_loss: 0.3598

Trial 7 Complete [00h 10m 51s]
val_accuracy: 0.8942499756813049

Best val_accuracy So Far: 0.9068333506584167
Total elapsed time: 01h 00m 56s

Search: Running Trial #8

Value             |Best Value So Far |Hyperparameter
128               |480               |units
0.3               |0.3               |dropout_rate
0.0048674         |0.00025615        |learning_rate
rmsprop           |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 89s 59ms/step - accuracy: 0.7743 - loss: 0.7758 - val_accuracy: 0.8579 - val_loss: 0.4298
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 83s 55ms/step - accuracy: 0.8442 - loss: 0.5007 - val_accuracy: 0.8675 - val_loss: 0.4678
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 82s 55ms/step - accuracy: 0.8541 - loss: 0.4972 - val_accuracy: 0.8752 - val_loss: 0.4709
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 84s 56ms/step - accuracy: 0.8558 - loss: 0.5173 - val_accuracy: 0.8766 - val_loss: 0.5076

Trial 8 Complete [00h 05m 38s]
val_accuracy: 0.8765833377838135

Best val_accuracy So Far: 0.9068333506584167
Total elapsed time: 01h 06m 34s

Search: Running Trial #9

Value             |Best Value So Far |Hyperparameter
416               |480               |units
0.2               |0.3               |dropout_rate
0.00201           |0.00025615        |learning_rate
rmsprop           |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 95s 62ms/step - accuracy: 0.8093 - loss: 0.6340 - val_accuracy: 0.8590 - val_loss: 0.4303
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 91s 60ms/step - accuracy: 0.8776 - loss: 0.3683 - val_accuracy: 0.8750 - val_loss: 0.3756
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 96s 64ms/step - accuracy: 0.8851 - loss: 0.3553 - val_accuracy: 0.8887 - val_loss: 0.3638
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 100s 67ms/step - accuracy: 0.8916 - loss: 0.3383 - val_accuracy: 0.8709 - val_loss: 0.4240
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 100s 67ms/step - accuracy: 0.8963 - loss: 0.3305 - val_accuracy: 0.8948 - val_loss: 0.3771
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 97s 65ms/step - accuracy: 0.9007 - loss: 0.3311 - val_accuracy: 0.8852 - val_loss: 0.4273

Trial 9 Complete [00h 09m 40s]
val_accuracy: 0.8948333263397217

Best val_accuracy So Far: 0.9068333506584167
Total elapsed time: 01h 16m 14s

Search: Running Trial #10

Value             |Best Value So Far |Hyperparameter
288               |480               |units
0.3               |0.3               |dropout_rate
0.0045758         |0.00025615        |learning_rate
adam              |adam              |optimizer

Epoch 1/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 94s 62ms/step - accuracy: 0.8019 - loss: 0.5953 - val_accuracy: 0.8695 - val_loss: 0.3607
Epoch 2/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 88s 59ms/step - accuracy: 0.8521 - loss: 0.4156 - val_accuracy: 0.8774 - val_loss: 0.3464
Epoch 3/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 87s 58ms/step - accuracy: 0.8570 - loss: 0.3940 - val_accuracy: 0.8793 - val_loss: 0.3404
Epoch 4/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 95s 63ms/step - accuracy: 0.8658 - loss: 0.3760 - val_accuracy: 0.8786 - val_loss: 0.3408
Epoch 5/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 91s 61ms/step - accuracy: 0.8720 - loss: 0.3594 - val_accuracy: 0.8820 - val_loss: 0.3602
Epoch 6/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 92s 61ms/step - accuracy: 0.8712 - loss: 0.3573 - val_accuracy: 0.8852 - val_loss: 0.3389
Epoch 7/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 92s 61ms/step - accuracy: 0.8775 - loss: 0.3375 - val_accuracy: 0.8870 - val_loss: 0.3364
Epoch 8/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 94s 63ms/step - accuracy: 0.8782 - loss: 0.3388 - val_accuracy: 0.8885 - val_loss: 0.3386
Epoch 9/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 96s 64ms/step - accuracy: 0.8858 - loss: 0.3148 - val_accuracy: 0.8808 - val_loss: 0.3729
Epoch 10/10
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 96s 64ms/step - accuracy: 0.8820 - loss: 0.3256 - val_accuracy: 0.8760 - val_loss: 0.3554

Trial 10 Complete [00h 15m 26s]
val_accuracy: 0.8884999752044678

Best val_accuracy So Far: 0.9068333506584167
Total elapsed time: 01h 31m 39s
"""

print("-"*100)

# --- 5. Retrieve the Best Model and Hyperparameters ---
print("--- 5. Retrieve the Best Model and Hyperparameters ---")

print("\n--- Random Search Results ---")
best_hps_random = tuner_random.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters found by Random Search:")
print(f"  Units in Dense layer: {best_hps_random.get('units')}")    # 480
print(f"  Dropout Rate: {best_hps_random.get('dropout_rate')}")     # 0.30000000000000004
print(f"  Learning Rate: {best_hps_random.get('learning_rate')}")   # 0.000256153969803113
print(f"  Optimiser: {best_hps_random.get('optimizer')}")           # adam

best_model_random = tuner_random.get_best_models(num_models=1)[0]
print(f"Best model's summary (from Random Search):")
best_model_random.summary()

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
│ dense (Dense)                        │ (None, 480)                 │         614,880 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 480)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           4,810 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,877,674 (10.98 MB)
 Trainable params: 619,690 (2.36 MB)
 Non-trainable params: 2,257,984 (8.61 MB)
313/313 ━━━━━━━━━━━━━━━━━━━━ 16s 50ms/step - accuracy: 0.8972 - loss: 0.2792  
"""

# Evaluate the best model on test data
loss_random, accuracy_random = best_model_random.evaluate(x_test, y_test)
print(f"Best model from Random Search - Test Loss: {loss_random:.4f}, Test Accuracy: {accuracy_random:.4f}")
# Best model from Random Search - Test Loss: 0.2769, Test Accuracy: 0.9010

"""
Observations:

Best Hyperparameters Found (from Trial #4):
- Units in Dense layer: 480
- Dropout Rate: 0.3
- Learning Rate: 0.000256153969803113 (approximately 2.56e-4)
- Optimizer: adam

Best Validation Accuracy: The highest val_accuracy achieved during the search was 0.9068 (from Trial #4).

Test Accuracy of the Best Model: The model configured with these best hyperparameters achieved a test accuracy of 0.9010. 
- This is very close to the validation accuracy, which suggests the model is generalising well and not just overfitting to the validation set.

Efficiency of Early Stopping: EarlyStopping callback saved significant computation time. 
- Many trials stopped after only a few epochs (e.g., Trial 1, 5, 6, 8) because their validation loss wasn't improving, preventing them from running for the full 10 epochs. 

"""
