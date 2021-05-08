import tensorflow as tf
import tensorflow_addons as tfa

class vgg16_model():
    def __init__(
              self, 
              IMG_SIZE,
              fine_tune=False,
              fine_tune_at=100,
              augmentation=True,
              dense_activation='relu',
              add_regularizer=True,
              add_dropout=True,
              class_num=1,
              weight_normalize=True,
              pred_activation='sigmoid',
              add_classification_head=True):


              self.IMG_SIZE = IMG_SIZE
              self.fine_tune = fine_tune
              self.fine_tune_at = fine_tune_at
              self.augmentation = augmentation
              self.dense_activation = dense_activation
              self.pred_activation = pred_activation
              self.add_regularizer = add_regularizer
              self.add_classification_head = add_classification_head
              self.add_dropout = add_dropout
              self.class_num = class_num
              self.weight_normalize = weight_normalize

              self.IMG_SHAPE = IMG_SIZE + (3,)

              self.base_model_v1 = tf.keras.applications.VGG16(input_shape=self.IMG_SHAPE,
                                               include_top=False,
                                               pooling='avg',
                                               weights='imagenet')

              self.base_model_v2 = tf.keras.applications.VGG16(input_shape=self.IMG_SHAPE,
                                               include_top=False,
                                               pooling='max',
                                               weights='imagenet')

              self.base_model_v3 = tf.keras.applications.VGG16(input_shape=self.IMG_SHAPE,
                                               include_top=False,
                                               pooling=None,
                                               weights='imagenet')

              self.data_augmentation = tf.keras.Sequential([
                        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
                        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
                        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
                    ])

              self.preprocess_input = tf.keras.applications.vgg16.preprocess_input
              self.prediction_layer = tf.keras.layers.Dense(self.class_num, activation=self.pred_activation)
              self.regularizer = tf.keras.layers.Lambda(lambda xz: tf.math.l2_normalize(xz, axis=1))

    def get_dense_layer(self, input, input_dim=512):
        if self.weight_normalize:
            dense = tfa.layers.WeightNormalization(tf.keras.layers.Dense(input_dim, activation=self.dense_activation))
            x = dense(input)
            return x
        else:
            return tf.keras.layers.Dense(input_dim, activation=self.dense_activation)(input)

    def add_dense_layer_with_dropout(self,input, input_dim=512, dropout_val=0.05):
        x = self.get_dense_layer(input, input_dim) 
        x = tf.keras.layers.Dropout(dropout_val)(x)
        x = self.get_dense_layer(x, input_dim//2)
        x = tf.keras.layers.Dropout(dropout_val*2)(x)
        x = self.get_dense_layer(x, input_dim//4)
        x = tf.keras.layers.Dropout(dropout_val*4)(x)
        x = self.get_dense_layer(x, input_dim//8)
        x = tf.keras.layers.Dropout(dropout_val*8)(x)
        
        return x

    def add_dense_layer(self,input, input_dim=512):
        x = self.get_dense_layer(input, input_dim)
        x = self.get_dense_layer(x, input_dim//2)
        x = self.get_dense_layer(x, input_dim//4)
        x = self.get_dense_layer(x, input_dim//8)
        return x

    def create_model_arch(self,
                 base_model, 
                 add_flat_layer=False):
        
        inputs = tf.keras.Input(shape=self.IMG_SHAPE)
        if self.augmentation:
            x = self.data_augmentation(inputs)
            x = self.preprocess_input(x)
        else:
            x = self.preprocess_input(inputs)

        x = base_model(x)
        if add_flat_layer:
            x = tf.keras.layers.Flatten()(x)
            if self.add_dropout:
                x = self.add_dense_layer_with_dropout(x, input_dim=1024)
            else:
                x = self.add_dense_layer(x, input_dim=1024)
        else:
            if self.add_dropout:
                x = self.add_dense_layer_with_dropout(x)
            else:
                x = self.add_dense_layer(x)

        if self.add_regularizer:
            x = self.regularizer(x) # L2 normalize embeddings
        
        if self.weight_normalize:
            output_layer = tfa.layers.WeightNormalization(self.prediction_layer)
            outputs = output_layer(x)
        else:
            outputs = self.prediction_layer(x)


        return tf.keras.Model(inputs, outputs)

    def vgg16_v1(self):
        if self.fine_tune:
            self.base_model_v1.trainable = True
            
               # Fine-tune from this layer onwards
               # Freeze all the layers before the `fine_tune_at` layer
            for layer in self.base_model_v1.layers[:self.fine_tune_at]:
                layer.trainable =  False
        else:
            self.base_model_v1.trainable = False

        return self.create_model_arch(self.base_model_v1, False)

    def vgg16_v2(self):
        if self.fine_tune:
            self.base_model_v2.trainable = True
               # Fine-tune from this layer onwards
               # Freeze all the layers before the `fine_tune_at` layer
            for layer in self.base_model_v2.layers[:self.fine_tune_at]:
                layer.trainable =  False
        else:
            self.base_model_v2.trainable = False

        return self.create_model_arch(self.base_model_v2, False)

    def vgg16_v3(self):
        if self.fine_tune:
            self.base_model_v3.trainable = True
               # Fine-tune from this layer onwards
               # Freeze all the layers before the `fine_tune_at` layer
            for layer in self.base_model_v3.layers[:self.fine_tune_at]:
                layer.trainable =  False
        else:
            self.base_model_v3.trainable = False

        return self.create_model_arch(self.base_model_v3, True)

        
