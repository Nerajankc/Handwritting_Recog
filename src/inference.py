import textwrap

from tensorflow.keras.layers import StringLookup
from tensorflow import keras
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.layers import StringLookup
import os
import pickle
import uuid

from main import save_image_names_to_text_files

max_len = 10
AUTOTUNE = tf.data.AUTOTUNE

# Fix the path to look for characters file in the src directory
char_file_path = os.path.join(os.path.dirname(__file__), "characters")
with open(char_file_path, "rb") as fp:   # Unpickling
    b = pickle.load(fp)
    print(b)

# Maping characaters to integers
char_to_num = StringLookup(vocabulary=b, mask_token=None)

#Maping integers back to original characters
num_to_chars = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# New function to run inference on a single image path and return results
def run_inference_on_image(image_path):
    # Create a unique output directory for this inference
    output_dir = os.path.join("../outputs", str(uuid.uuid4()))
    
    # Process the image and get output paths
    output_dir, image_paths = save_image_names_to_text_files(image_path, output_dir)
    
    # Get all processed images from the directory
    t_images = image_paths
    print(f"Processing {len(t_images)} segmented images")
    
    # Clear previous results
    pred_test_text.clear()
    
    # Prepare the images for inference
    inf_images = prepare_test_images(t_images)
    
    # Run inference on the prepared images
    for batch in inf_images.take(3):
        batch_images = batch["image"]
        print(batch_images.shape)
        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        pred_test_text.append(pred_texts)
    
    # Combine results
    flat_list = [item for sublist in pred_test_text for item in sublist]
    print(flat_list)
    
    sentence = ' '.join(flat_list)
    wrapped_text = textwrap.fill(sentence, width=80)
    print(wrapped_text)
    
    # Format results
    result = {
        "prediction": flat_list,
        "formatted_text": wrapped_text,
        "output_directory": output_dir
    }
    
    return result

def distortion_free_resize(image, img_size):
  w, h = img_size
  image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

  # Check tha amount of padding needed to be done.
  pad_height = h - tf.shape(image)[0]
  pad_width = w - tf.shape(image)[1]

  # only necessary if you want to do same amount of padding on both sides.
  if pad_height % 2 != 0:
    height = pad_height // 2
    pad_height_top = height +1
    pad_height_bottom = height
  else:
    pad_height_top = pad_height_bottom = pad_height // 2

  if pad_width % 2 != 0:
    width = pad_width // 2
    pad_width_left = width + 1
    pad_width_right = width
  else:
    pad_width_left = pad_width_right = pad_width // 2

  image = tf.pad(
      image, paddings=[
          [pad_height_top, pad_height_bottom],
          [pad_width_left, pad_width_right],
          [0, 0],
      ],
  )
  image = tf.transpose(image, perm=[1,0,2])
  image = tf.image.flip_left_right(image)
  return image

# Testing inference images
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


def preprocess_image(image_path, img_size=(image_width, image_height)):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_png(image, 1)
  image = distortion_free_resize(image, img_size)
  image = tf.cast(image, tf.float32) / 255.0
  return image

def process_images_2(image_path):
  image = preprocess_image(image_path)
  # label = vectorize_label(label)
  return {"image": image}
  
def prepare_test_images(image_paths):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths)).map(
    process_images_2, num_parallel_calls=AUTOTUNE
  )

  # return dataset
  return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


class CTCLayer(keras.layers.Layer):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.loss_fn = tf.keras.backend.ctc_batch_cost  

  def call(self, y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = self.loss_fn(y_true, y_pred, input_length, label_length)
    self.add_loss(loss)

    return y_pred

custom_objects = {"CTCLayer": CTCLayer}
# Fix the path to the model file
model_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "epoch50_our_ocr_pred_model.h5")
reconstructed_model = keras.models.load_model(model_file_path, custom_objects=custom_objects)
prediction_model = keras.models.Model(reconstructed_model.get_layer(name="image").output, reconstructed_model.get_layer(name="dense2").output)
pred_test_text = []

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]

    output_text = []

    for res in results:
      res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
      res = tf.strings.reduce_join(num_to_chars(res)).numpy().decode("utf-8")
      output_text.append(res)

    return output_text

# If this module is imported, the following will not execute
if __name__ == "__main__":
    # Example of running inference on a single image if needed
    pass

