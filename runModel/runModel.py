import tensorflow as tf
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(('chatbotpersonal.ddns.net', 5555))

s.listen()

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

model = tf.keras.models.load_model('saved_model')

while True:
    client, address = s.accept()
    print(f'connected to {address}')
    client.send('You are connected'.encode())
    client.close()