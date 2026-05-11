# Dependencies
import tensorflow as tf

class Model2D(tf.keras.Model):
    # default

    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3    # T1w + T2w + FLAIR (multi-modal)
    inputs = []
    outputs = []

    def __init__(self, IMG_HEIGHT: int = 256, IMG_WIDTH: int = 256, IMG_CHANNELS: int = 3):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS

        # Define model

        self.inputs = tf.keras.layers.Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        
        # normalize input
        norm_inputs = tf.keras.layers.Lambda(lambda x: x/255.0)(self.inputs)

        # Downsampling

        # Layer 1
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(norm_inputs)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        # Layer 2
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        # Layer 3
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        # Layer 4
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

        # Layer 5
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

        # Upsampling

        # Layer 6
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

        # Layer 7
        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
        c7 = tf.keras.layers.Conv2D(256, (3, 3),activation="relu", kernel_initializer="he_normal", padding="same")(c7)

        # Layer 8
        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
        u8 = tf.keras.layers.concatenate([u8, c2]) 
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

        # Layer 9
        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8) 
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

        # Output Layer (mask)
        self.outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

    def initializeModel(self):
        self.model = tf.keras.Model(inputs = [self.inputs], outputs = [self.outputs])
        self.model.compile(optimizer="adam", loss=tf.keras.losses.Dice(), metrics=["accuracy"])

    def summarize(self):
        self.model.summary()

def main():
    model = Model2D(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)  # T1w + T2w + FLAIR
    model.initializeModel()
    model.summarize()

if __name__ == "__main__":
    main()


        

