import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module


def load_data(data_sampling_percentage=0.005,batch_size=32):
    # The use of data_sampling_percentage allows us to use a subset of the data and not the entire dataset in order to avoid out of memory errors

    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data: normalize images
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Convert the datasets to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Calculate subset size for train and test datasets
    train_subset_size = int(len(train_images) * data_sampling_percentage)
    test_subset_size = int(len(test_images) * data_sampling_percentage)

    # Shuffle and subset data
    train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).take(train_subset_size)
    test_dataset = test_dataset.shuffle(buffer_size=len(test_images)).take(test_subset_size)

    # Batch data
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Optimize datasets for performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    logger.info("Created data generators with batch size: %s", batch_size)
    
    return train_dataset, test_dataset, train_subset_size, test_subset_size
