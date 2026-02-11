
import tensorflow as tf
import numpy as np
from data import Person, person_from_keypoints_with_scores

class Movenet(object):
    """Movenet pose estimation model."""

    def __init__(self, model_name):
        # model_name is expected to be 'movenet_thunder' or similar, 
        # but we look for the file 'movenet_thunder.tflite' or similar.
        # The proprocessing.py downloads 'movenet_thunder.tflite'.
        
        if model_name == 'movenet_thunder':
             model_path = 'movenet_thunder.tflite'
             self._input_size = 256
        elif model_name == 'movenet_lightning':
             model_path = 'movenet_lightning.tflite' # Assumption, might not be used
             self._input_size = 192
        else:
             # Fallback or assume path is provided directly if needed, 
             # but for this specific code, it passes 'movenet_thunder'
             model_path = 'movenet_thunder.tflite' 
             self._input_size = 256

        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        
    def detect(self, input_image, reset_crop_region=False):
        """
        Runs detection on an input image.
        
        Args:
          input_image: A [height, width, 3] numpy array representing the input image.
          reset_crop_region: Not used in this basic implementation but kept for API compatibility.
        
        Returns:
          A Person entity.
        """
        
        # Resize and pad the image to keep aspect ratio and fit model input.
        # For simplicity in this basic implementation, we might just resize directly 
        # or use tf.image.resize_with_pad which is robust.
        
        image_height, image_width, _ = input_image.shape
        
        input_tensor = tf.expand_dims(input_image, axis=0)
        input_tensor = tf.image.resize_with_pad(input_tensor, self._input_size, self._input_size)
        
        # MoveNet expects int32 input for some versions, but float32 for others.
        # The URL in proprocessing.py is float16 model. 
        # Let's check input dtype from details usually, but for now we cast to int32 
        # as standard MoveNet examples often do, or keep as is if input expects float.
        # However, tf.image.resize_with_pad returns float32.
        
        if self._input_details[0]['dtype'] == np.uint8:
             input_tensor = tf.cast(input_tensor, dtype=tf.uint8)
        elif self._input_details[0]['dtype'] == np.int32:
             input_tensor = tf.cast(input_tensor, dtype=tf.int32)
        
        self._interpreter.set_tensor(self._input_details[0]['index'], input_tensor.numpy())
        self._interpreter.invoke()

        keypoints_with_scores = self._interpreter.get_tensor(self._output_details[0]['index'])[0][0]
        
        return person_from_keypoints_with_scores(
            keypoints_with_scores,
            image_height,
            image_width)
