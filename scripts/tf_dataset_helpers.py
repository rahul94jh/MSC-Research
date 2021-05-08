import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API


class read_tfrec_data:
    def __init__(self,
                 FILE_PATH,
                 TARGET_SIZE=[180, 180], 
                 MODE=2, 
                 BATCH_SIZE=32, 
                 SHUFFLE_BUFFER=10000,
                 CACHE=True,
                 REPEAT=True,
                 PREFETCH=True,
                 VALIDATION_SPLIT=0.3,
                 TESTING_SPLIT=0.5):

        self.FILE_PATH = FILE_PATH
        self.TARGET_SIZE = TARGET_SIZE
        self.MODE = MODE
        self.BATCH_SIZE = BATCH_SIZE
        self.SHUFFLE_BUFFER = SHUFFLE_BUFFER
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.TESTING_SPLIT = TESTING_SPLIT
        self.CACHE = CACHE
        self.REPEAT = REPEAT
        self.PREFETCH = PREFETCH

    def get_tfrec_files(self):
          filenames = tf.io.gfile.glob(self.FILE_PATH + "*.tfrec")
          split = int(len(filenames) * self.VALIDATION_SPLIT)
          training_filenames = filenames[split:]
          intermediate_filenames = filenames[:split]
          test_split = int(len(intermediate_filenames) * self.TESTING_SPLIT)
          validation_filenames = intermediate_filenames[test_split:]
          testing_filenames = intermediate_filenames[:test_split]
          print("Pattern matches {} data files. Splitting dataset into {} training files , {} validation files and {} test files".format(len(filenames), len(training_filenames), len(validation_filenames), len(testing_filenames)))
          return  filenames, training_filenames, validation_filenames, testing_filenames
    
    def read_tfrecord(self,example):
            features = {
              "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
              "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
              "text": tf.io.FixedLenFeature([], tf.string),
        
            # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
              "label":         tf.io.FixedLenFeature([], tf.string),  # one bytestring
              "size":          tf.io.FixedLenFeature([2], tf.int64)  # two integers
            }
            # decode the TFRecord
            example = tf.io.parse_single_example(example, features)
    
           # FixedLenFeature fields are now ready to use: exmple['size']
        # VarLenFeature fields require additional sparse_to_dense decoding
    
            image = tf.image.decode_jpeg(example['image'], channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.reshape(image, [*self.TARGET_SIZE, 3])
    
            class_num = example['class']
            text = example['text']
            label  = example['label']
            height = example['size'][0]
            width  = example['size'][1]
            return image, text, class_num, label, height, width
       
    def load_dataset(self,filenames):
            # read from TFRecords. For optimal performance, read from multiple
            # TFRecord files at once and set the option experimental_deterministic = False
            # to allow order-altering optimizations.

            option_no_order = tf.data.Options()
            option_no_order.experimental_deterministic = False

            dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
            dataset = dataset.with_options(option_no_order)
            dataset = dataset.map(self.read_tfrecord, num_parallel_calls=AUTO)
            if self.MODE==0: 
                dataset = dataset.map(lambda image, text, class_num, label, height, width: (image, class_num))
            elif self.MODE==1:
                dataset = dataset.map(lambda image, text, class_num, label, height, width: (text, class_num))
            else:
                dataset = dataset.map(lambda image, text, class_num, label, height, width: ((image,text), class_num))
            return dataset

    def get_batched_dataset(self,filenames, train=False):
            dataset = self.load_dataset(filenames)
            if self.CACHE:
                dataset = dataset.cache() # This dataset fits in RAM
            if train:
                # Best practices for Keras:
                # Training dataset: repeat then batch
                # Evaluation dataset: do not repeat
                dataset = dataset.shuffle(self.SHUFFLE_BUFFER)
                if self.REPEAT:
                    dataset = dataset.repeat()
            dataset = dataset.batch(self.BATCH_SIZE)
            if self.PREFETCH:
                dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
            # should shuffle too but this dataset was well shuffled on disk already
            return dataset
            # source: Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets