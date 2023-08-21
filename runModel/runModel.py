import tensorflow as tf
import numpy as np
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(('chatbotpersonal.ddns.net', 5555))
#here in the video they used 127.0.0.1 as it is for localhost

s.listen()

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

model = tf.keras.models.load_model('saved_model/my_model')

while True:
    client, address = s.accept()
    print(f'connected to {address}')
    client.send('You are connected'.encode())
    client.close()