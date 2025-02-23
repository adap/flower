import warnings
try:
    import tensorflow as tf
except ImportError:
    warnings.warn("tensorflow is not installed. CustomWandbCallback will not work.")


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def __init__(self, node_i):
        self.node_i = node_i
        self.node_i_name = f"node{node_i}"

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        log_dict = {
            f"{self.node_i_name}_epoch": epoch,
            f"{self.node_i_name}_loss": logs["loss"],
            f"{self.node_i_name}_accuracy": logs["accuracy"],
            f"{self.node_i_name}_val_loss": logs["val_loss"],
            f"{self.node_i_name}_val_accuracy": logs["val_accuracy"],
        }

        import wandb

        wandb.log(log_dict)
