# Dogs vs. Cats Classification

This repository contains a Convolutional Neural Network (CNN) model for classifying images of dogs and cats. The dataset is sourced from Kaggle, and the model is built using TensorFlow and Keras. The model achieves over 80% accuracy on the validation set after 10 epochs of training.

## Dataset

The dataset used in this project is the "Dogs vs. Cats" dataset available on Kaggle. It consists of 2500 images of dogs and cats.

- [Dataset URL](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

## Setup and Requirements

To run the code in this repository, you need the following:

- Kaggle API key (`kaggle.json`)
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib

### Steps to Download and Prepare the Dataset

1. **Download the Dataset:**

    Ensure you have the Kaggle API key in your working directory.

    ```bash
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    ```

    Download the dataset:

    ```bash
    !kaggle datasets download -d salader/dogs-vs-cats
    ```

2. **Unzip the Dataset:**

    ```python
    import zipfile

    zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
    zip_ref.extractall('/content')
    zip_ref.close()
    ```

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

1. Conv2D
2. MaxPooling2D
3. BatchNormalization
4. Flatten
5. Dense
6. Dropout

### Model Summary

```plaintext
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_3 (Conv2D)           (None, 254, 254, 32)      896       
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 127, 127, 32)      0         
 g2D)                                                            
                                                                 
 batch_normalization (Batch  (None, 127, 127, 32)      128       
 Normalization)                                                  
                                                                 
 conv2d_4 (Conv2D)           (None, 125, 125, 64)      18496     
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 62, 62, 64)        0         
 g2D)                                                            
                                                                 
 batch_normalization_1 (Bat  (None, 62, 62, 64)        256       
 chNormalization)                                                
                                                                 
 conv2d_5 (Conv2D)           (None, 60, 60, 128)       73856     
                                                                 
 max_pooling2d_5 (MaxPoolin  (None, 30, 30, 128)       0         
 g2D)                                                            
                                                                 
 batch_normalization_2 (Bat  (None, 30, 30, 128)       512       
 chNormalization)                                                
                                                                 
 flatten_1 (Flatten)         (None, 115200)            0         
                                                                 
 dense_3 (Dense)             (None, 128)               14745728  
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_4 (Dense)             (None, 64)                8256      
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_5 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 14848193 (56.64 MB)
Trainable params: 14847745 (56.64 MB)
Non-trainable params: 448 (1.75 KB)
_________________________________________________________________
```

## Training the Model

The model is trained for 10 epochs with a batch size of 32.

```python
history = model.fit(train_ds, epochs = 10, validation_data = validation_ds)
```

### Training and Validation Results

```plaintext
Epoch 1/10
625/625 [==============================] - 68s 93ms/step - loss: 2.2282 - accuracy: 0.5980 - val_loss: 0.6393 - val_accuracy: 0.6390
Epoch 2/10
625/625 [==============================] - 55s 88ms/step - loss: 0.5631 - accuracy: 0.7104 - val_loss: 0.5365 - val_accuracy: 0.7212
...
Epoch 10/10
625/625 [==============================] - 57s 91ms/step - loss: 0.0965 - accuracy: 0.9661 - val_loss: 0.6823 - val_accuracy: 0.8088
```

## Reducing Overfitting

Several techniques can be used to reduce overfitting, including:

- Adding more data
- Data augmentation
- L1/L2 regularization
- Dropout (already applied)
- Batch normalization (already applied)
- Reducing model complexity

## Testing the Model

To test the model on a new image:

1. Read and preprocess the image:

    ```python
    import cv2
    import matplotlib.pyplot as plt

    test_img = cv2.imread('/content/cat.jpg')
    plt.imshow(test_img)
    test_img = cv2.resize(test_img, (256, 256))
    test_input = test_img.reshape((1, 256, 256, 3))
    ```

2. Predict the class:

    ```python
    model.predict(test_input)
    ```

    Example output:

    ```plaintext
    array([[1.]], dtype=float32)
    ```

## License

This project uses the dataset provided by Kaggle under its terms of service. Ensure to adhere to the dataset's license for any further usage or distribution.

## References

- [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

## Acknowledgments

This is a practice project to brush up my Deep Learning skills.

---

