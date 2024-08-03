import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Input, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet(input_shape, num_classes, num_blocks=[2, 2, 2, 2]):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    for i, block_num in enumerate(num_blocks):
        for j in range(block_num):
            if i == 0 and j == 0:
                pass
            else:
                strides = 2 if j == 0 else 1
                filters = 64 * 2**i
                x = residual_block(x, filters, stride=strides)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (100, 100, 1)
num_classes = 2
model = build_resnet(input_shape, num_classes)
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
report = classification_report(y_test, y_pred_classes)
print(report)
cm = confusion_matrix(y_test, y_pred_classes)
print(cm)


# Update HTML report with results
html_report = f'''
<!DOCTYPE html>
<html>
<head>
  <style>
    ... (styling)
  </style>
</head>
<body>
  <h2>Classification Report</h2>
  <table>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision (Sparse/Dense)</th>
      <th>Recall (Sparse/Dense)</th>
      <th>F1-score (Sparse/Dense)</th>
      <th>SVM Accuracy as FE</th>
      <th>RF Accuracy as FE</th>
      <th>SVM Precision as FE</th>
      <th>RF Precision as FE</th>
      <th>SVM Recall as FE</th>
      <th>RF Recall as FE</th>
    </tr>
    <tr>
      <td>ResNet</td>
      <td>{test_acc:.4f}</td>
      <td>{report['1']['precision']:.4f}/{report['0']['precision']:.4f}</td>
      <td>{report['1']['recall']:.4f}/{report['0']['recall']:.4f}</td>
      <td>{report['1']['f1-score']:.4f}/{report['0']['f1-score']:.4f}</td>
      <td></td>  # Add SVM and RF results later
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </table>
</body>
</html>
'''

# Write the HTML report to a file
with open("classification_report.html", "w") as file:
  file.write(html_report)

# Display the HTML report as output
from IPython.display import display, HTML
display(HTML(html_report))

# Save the trained model
model.save('resnet_model.h5')
