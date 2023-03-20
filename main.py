import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json


class ImageClassifier:
    def __init__(self, train_csv_path, test_size=0.2, img_size=(224, 224), batch_size=32, epochs=5, lr=0.0001):
        self.train_csv_path = train_csv_path
        self.test_size = test_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.train_df = None
        self.test_df = None
        self.train_datagen = None
        self.test_datagen = None
        self.train_generator = None
        self.test_generator = None
        self.model = None
        self.history = None
        self.f1_score = None
        self.cm = None
        self.results = None
        
    def load_data(self):
        df = pd.read_csv(self.train_csv_path, index_col=0)
        df['label'] = df['label'].astype(str)
        self.train_df = df.sample(frac=1 - self.test_size, random_state=0)
        self.test_df = df.drop(self.train_df.index)
        
    def create_generators(self):
        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_datagen.flow_from_dataframe(
            self.train_df,
            x_col='path_img',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        self.test_generator = self.test_datagen.flow_from_dataframe(
            self.test_df,
            x_col='path_img',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(8, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr), metrics=['accuracy'])
        
    def train_model(self):
        checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.history = self.model.fit(self.train_generator, epochs=self.epochs, validation_data=self.test_generator, callbacks=[checkpoint])

    def evaluate_model(self):
        self.model.load_weights('model.h5')
        y_pred = self.model.predict(self.test_generator)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = self.test_generator.classes
        self.f1_score = f1_score(y_true, y_pred, average='micro')
        self.cm = confusion_matrix(y_true, y_pred)
        self.results = self.model.evaluate(self.test_generator)

    def plot_history(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def plot_confusion_matrix(self):
        plt.figure(figsize=(10, 10))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def save_results(self):
        with open('results.json', 'w') as f:
            json.dump({'f1_score': self.f1_score, 'loss': self.results[0], 'accuracy': self.results[1]}, f)
    

if __name__ == '__main__':
    classifier = ImageClassifier(train_csv_path='train.csv')
    #classifier.load_data()
    #classifier.create_generators()
    #classifier.create_model()
    #classifier.train_model()
    #classifier.evaluate_model()
    #classifier.plot_history()
    #classifier.plot_confusion_matrix()
    #classifier.save_results()

#leer model.h5 para predecir otro conjunto de datos

model = tf.keras.models.load_model('model.h5')
test_df = pd.read_csv('test.csv', index_col=0)

#predecir los label de test.csv que contine solo la img_path 
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='path_img',
    y_col=None,
    target_size=(224, 224),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

#guardar los resultados en un csv
test_df['label'] = y_pred
test_df.to_csv('results.csv')

#convertir a json con el key como el indicd del df y el value como el label en int
results = test_df['label'].to_dict()
results = {str(k): int(v) for k, v in results.items()}
final_dict = {"target": results}

#guardar el final_dict en un json
with open('predictions.json', 'w') as f:
    json.dump(final_dict, f)
    



