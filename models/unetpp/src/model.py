import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf

class UNetPlusPlus(tf.keras.Model):
    """
    Keras implementation of 2-D UNet++ (Nested U-Net) for multi-modal brain MRI segmentation.
    Supports input size of (256, 256, 3).
    """

    def __init__(self, IMG_HEIGHT: int = 256, IMG_WIDTH: int = 256, IMG_CHANNELS: int = 3):
        super().__init__()
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS

        # Input Layer
        self.inputs = tf.keras.layers.Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        
        # ─── HELPER BLOCK FUNCTION ───────────────────────────────────────────
        def conv_block(x, filters, name):
            c = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name=f"{name}_conv1")(x)
            c = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(c)
            c = tf.keras.layers.Dropout(0.1)(c)
            c = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name=f"{name}_conv2")(c)
            c = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(c)
            return c

        def upsample_layer(x, name):
            return tf.keras.layers.Conv2DTranspose(x.shape[-1], (2, 2), strides=(2, 2), padding="same", name=name)(x)

        # ─── ENCODER (Backbone, Col 0) ───────────────────────────────────────
        # Level 0 -> (256, 256, 32)
        X_00 = conv_block(self.inputs, 32, "x00")
        p0 = tf.keras.layers.MaxPooling2D((2, 2), name="p0")(X_00)

        # Level 1 -> (128, 128, 64)
        X_10 = conv_block(p0, 64, "x10")
        p1 = tf.keras.layers.MaxPooling2D((2, 2), name="p1")(X_10)

        # Level 2 -> (64, 64, 128)
        X_20 = conv_block(p1, 128, "x20")
        p2 = tf.keras.layers.MaxPooling2D((2, 2), name="p2")(X_20)

        # Level 3 -> (32, 32, 256)
        X_30 = conv_block(p2, 256, "x30")
        p3 = tf.keras.layers.MaxPooling2D((2, 2), name="p3")(X_30)

        # Level 4 (Bottleneck) -> (16, 16, 512)
        X_40 = conv_block(p3, 512, "x40")

        # ─── NESTED PATHWAYS (Col 1) ─────────────────────────────────────────
        # X_01 receives X_00, and upsampled X_10
        u01 = upsample_layer(X_10, "u10_to_01")
        X_01 = conv_block(tf.keras.layers.concatenate([X_00, u01]), 32, "x01")

        # X_11 receives X_10, and upsampled X_20
        u11 = upsample_layer(X_20, "u20_to_11")
        X_11 = conv_block(tf.keras.layers.concatenate([X_10, u11]), 64, "x11")

        # X_21 receives X_20, and upsampled X_30
        u21 = upsample_layer(X_30, "u30_to_21")
        X_21 = conv_block(tf.keras.layers.concatenate([X_20, u21]), 128, "x21")

        # X_31 receives X_30, and upsampled X_40
        u31 = upsample_layer(X_40, "u40_to_31")
        X_31 = conv_block(tf.keras.layers.concatenate([X_30, u31]), 256, "x31")

        # ─── NESTED PATHWAYS (Col 2) ─────────────────────────────────────────
        # X_02 receives X_00, X_01, and upsampled X_11
        u02 = upsample_layer(X_11, "u11_to_02")
        X_02 = conv_block(tf.keras.layers.concatenate([X_00, X_01, u02]), 32, "x02")

        # X_12 receives X_10, X_11, and upsampled X_21
        u12 = upsample_layer(X_21, "u21_to_12")
        X_12 = conv_block(tf.keras.layers.concatenate([X_10, X_11, u12]), 64, "x12")

        # X_22 receives X_20, X_21, and upsampled X_31
        u22 = upsample_layer(X_31, "u31_to_22")
        X_22 = conv_block(tf.keras.layers.concatenate([X_20, X_21, u22]), 128, "x22")

        # ─── NESTED PATHWAYS (Col 3) ─────────────────────────────────────────
        # X_03 receives X_00, X_01, X_02, and upsampled X_12
        u03 = upsample_layer(X_12, "u12_to_03")
        X_03 = conv_block(tf.keras.layers.concatenate([X_00, X_01, X_02, u03]), 32, "x03")

        # X_13 receives X_10, X_11, X_12, and upsampled X_22
        u13 = upsample_layer(X_22, "u22_to_13")
        X_13 = conv_block(tf.keras.layers.concatenate([X_10, X_11, X_12, u13]), 64, "x13")

        # ─── NESTED PATHWAYS (Col 4 - Final Decoder node) ────────────────────
        # X_04 receives X_00, X_01, X_02, X_03, and upsampled X_13
        u04 = upsample_layer(X_13, "u13_to_04")
        X_04 = conv_block(tf.keras.layers.concatenate([X_00, X_01, X_02, X_03, u04]), 32, "x04")

        # Final Output Conv
        self.outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="final_sigmoid")(X_04)

    def initializeModel(self):
        self.model = tf.keras.Model(inputs=[self.inputs], outputs=[self.outputs])

    def summarize(self):
        self.model.summary()

def main():
    model = UNetPlusPlus(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
    model.initializeModel()
    model.summarize()

if __name__ == "__main__":
    main()