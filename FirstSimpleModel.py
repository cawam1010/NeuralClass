import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import graphviz
import pydot
import matplotlib.pyplot as plt

batchSize = 16

#Needs target in final column to the right# Should fix that eventually.
dataframe = pd.read_csv('ConstructedDataSetV9.csv')
dataframe.shape
print (dataframe.head())

# 10% val 30% test 206 rows currently total.
val_dataframe = dataframe.sample(frac=0.1, random_state=1337)
smallerdataframe = dataframe.drop(val_dataframe.index)
test_dataframe = smallerdataframe.sample(frac=0.3, random_state=1337)
train_dataframe = dataframe.drop(test_dataframe.index)
dataLabelStrings = (train_dataframe.keys())

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

# Simple make a data frame remove our target from the dataset. 
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("IncOrDec")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)
test_ds = dataframe_to_dataset(test_dataframe)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
test_ds = test_ds.batch(32)


for x, y in train_ds.take(1):
    print("Input:", x)
    print("incOrDec:", y)
    target=y
    
    
def encodeMyFeature(indidualFeature, name, dataset):
    # Normalization the data
    normalizer = Normalization()
    # Pull out a data set for each feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # built in code to find the statistics of the data
    normalizer.adapt(feature_ds)
    encodedFeature = normalizer(indidualFeature)
    return encodedFeature


#Grab our inputs
allMyInputs = []
for x in dataLabelStrings:
    if(x != "IncOrDec"):
        allMyInputs.append(keras.Input(shape=(1,), name=x))
    else:
        print ("Do not input the output")


#Encode our data. 
encoded_array = []
iii = 0
while (iii < len(dataLabelStrings)-1):
    print (dataLabelStrings[iii])
    encodedFeat = (encodeMyFeature(allMyInputs[iii],dataLabelStrings[iii],train_ds))
    encoded_array.append(encodedFeat)
    iii = iii + 1

##Encoded array of features.
allFeatures = layers.concatenate(encoded_array)

#Model Parameters
learning_rate = 0.0001
dropout_rate = 0.3
num_epochs = 4000
num_classes = 2
hidden_units = [32, 32]
encoding_size = 32
num_trees = 15
depth = 15
used_features_rate = .7
num_classes = len(dataLabelStrings)

def runMyModel(model):

#Tried some other optimizers and loss this seemed to work best.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("Start training my model")
    history = model.fit(train_ds, epochs=num_epochs,validation_data=val_ds)
    print("Model training finished")
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    _, accuracy = model.evaluate(test_ds, verbose=0)
    # summarize history for accuracy
    plt.plot(accuracy)
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    
def createSimpleModel(allFeatures,allInputs):
    
    for units in hidden_units:
        allFeatures = layers.Dense(units=500,activation='relu')(allFeatures)
        allFeatures = layers.Dropout(dropout_rate)(allFeatures)
        allFeatures = layers.Dense(units=100,activation='softmax')(allFeatures)
        allFeatures = layers.Dense(units=50,activation='softmax')(allFeatures)
        allFeatures = layers.ReLU()(allFeatures)
        allFeatures = layers.Dropout(dropout_rate)(allFeatures)
        

    outputs = layers.Dense(units=4, activation="softmax")(allFeatures)
    model = keras.Model(inputs=allInputs, outputs=outputs)
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    return model

#Create / Run Model.
model = createSimpleModel(allFeatures,allMyInputs)
keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
runMyModel(model)
#print(model.summary())
