import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf

class Model2D(tf.keras.Model):

    def __init__(self, IMG_HEIGHT: int = 256, IMG_WIDTH: int = 256, IMG_CHANNELS: int = 3):
        super().__init__()
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS

        # Define model inputs
        self.inputs = tf.keras.layers.Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        norm_inputs = self.inputs

        # =================================================================
        # DOWNSAMPLING (Encoder)
        # =================================================================

        # Layer 1 -> (256, 256, 64)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(norm_inputs)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) # -> (128, 128, 64)

        # Layer 2 -> (128, 128, 128)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) # -> (64, 64, 128)

        # Layer 3 -> (64, 64, 256)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) # -> (32, 32, 256)

        # Layer 4 -> (32, 32, 512)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4) # -> (16, 16, 512)

        # Layer 5 (Bottleneck) -> (16, 16, 1024)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
        c5 = tf.keras.layers.BatchNormalization()(c5)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)

        # =================================================================
        # UPSAMPLING (Decoder) with Shape Synchronization
        # =================================================================

        # Level 6: Target shape is c4 (32, 32)
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5) # -> (32, 32)
        u6 = self._match_and_pad(u6, c4)                                                    # Already (32, 32)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

        # Level 7: Target shape is c3 (64, 64)
        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6) # -> (64, 64)
        u7 = self._match_and_pad(u7, c3)                                                    # Already (64, 64)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

        # Level 8: Target shape is c2 (128, 128)
        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7) # -> (128, 128)
        u8 = self._match_and_pad(u8, c2)                                                    # Already (128, 128)
        u8 = tf.keras.layers.concatenate([u8, c2]) 
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

        # Level 9: Target shape is c1 (256, 256)
        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)  # -> (256, 256)
        u9 = self._match_and_pad(u9, c1)                                                    # Already (256, 256)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

        # Output Segmented Matrix
        self.outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

    def _match_and_pad(self, moving_tensor, target_tensor):
        """Helper to calculate and execute edge padding dynamically if shapes mismatch."""
        moving_shape = moving_tensor.shape
        target_shape = target_tensor.shape
        
        height_diff = target_shape[1] - moving_shape[1]
        width_diff = target_shape[2] - moving_shape[2]
        
        if height_diff > 0 or width_diff > 0:
            return tf.keras.layers.ZeroPadding2D(padding=((0, height_diff), (0, width_diff)))(moving_tensor)
        return moving_tensor

    def initializeModel(self):
        self.model = tf.keras.Model(inputs=[self.inputs], outputs=[self.outputs])

    def summarize(self):
        self.model.summary()

def main():
    model = Model2D(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
    model.initializeModel()
    model.summarize()

if __name__ == "__main__":
    main()
