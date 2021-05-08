import visualization_helpers as vh
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

def create_mobilenet_v2_model(IMG_SIZE, 
                              fine_tune=False,
                              fine_tune_at=100, 
                              augmentation=False, 
                              dense_size=512, 
                              dense_activation='relu',
                              add_classification_head=True):
  IMG_SHAPE = IMG_SIZE + (3,)
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

  # Freeze the base_model
  if not fine_tune:
      base_model.trainable = False
  else:
    base_model.trainable = True
    # Fine-tune from this layer onwards
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

  data_augmentation = tf.keras.Sequential([
     tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
     tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
     tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
     tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
  ])

  preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input  # RESCALE PIXEL TO [-1,1]
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  dense_layer = tf.keras.layers.Dense(dense_size, activation=dense_activation)

  prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

  inputs = tf.keras.Input(shape=(IMG_SHAPE))
  if augmentation:
      x = data_augmentation(inputs)
      x = preprocess_input(x)
      x = base_model(x, training=False)
  else:
      x = preprocess_input(inputs)
      x = base_model(x, training=False) # use training=False as base model contains a BatchNormalization layer

  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = dense_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)

  if add_classification_head:
      outputs = prediction_layer(x)
  else:
      outputs = x

  model = tf.keras.Model(inputs, outputs)

  return model 


def compile_train_model(
    model,
    train_ds,
    val_ds,
    steps_per_epoch,
    validation_steps,
    save_model=False,
    model_save_path='',
    EPOCHS=20,
    LOSS = keras.losses.BinaryCrossentropy(from_logits=True),
    OPTIMIZER = tf.keras.optimizers.Adam(lr=0.0001),
    METRICS = [
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'), 
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
     ],
    CALLBACKS = [EarlyStopping(monitor='val_prc', verbose=1,patience=10,mode='max',restore_best_weights=True)]
    ):

  model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=METRICS
    )

  history = model.fit(
    train_ds,
    steps_per_epoch = steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    verbose=1,
    callbacks=CALLBACKS)
  
  if save_model:
    if len(model_save_path)>0:
      model.save(model_save_path)
    else:
      print('unable to save model...')
  
  return model, history


def get_metrics(y_test, y_pred):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_pred)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_pred)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_pred)
    print('F1 score: %f' % f1)
    # ROC AUC
    auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

def predict_on_test_vis(model, ds, BATCH_SIZE=32, N=1):
  test_data = ds.take(N)
  test_image=[]
  y_true=[]
  
  for images, labels in test_data:
    for i in range(BATCH_SIZE):
      test_image.append(images[i].numpy())
      y_true.append(labels[i].numpy())


  predicted_scores  = model.predict(test_data)
  y_pred = (predicted_scores > 0.5).astype("int32")
  y_pred = y_pred.reshape(-1)
  get_metrics(y_true, y_pred)
  vh.plot_cm(y_true, predicted_scores)
  print()
  
  return test_image, y_true, y_pred, predicted_scores

def prediction_image_ds(model, test_dataset, class_names):
    #Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    y_pred = tf.where(predictions < 0.5, 0, 1)
    
    #print('Predictions:\n', predictions.numpy())
    #print('Labels:\n', label_batch) 

    get_metrics(label_batch, y_pred)
    vh.plot_cm(label_batch, predictions)

    plt.figure(figsize=(10, 10))
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(image_batch[i].astype("uint8"))
      plt.title(class_names[y_pred[i]])
      plt.axis("off") 

    return label_batch, y_pred