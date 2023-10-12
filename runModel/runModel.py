import tensorflow as tf
import threading as thread
import time
import pyaudio
import wave
import pathlib
import os

txtname = ["no.txt", "hi.txt"]
model = tf.keras.models.load_model('saved_model/my_model')

p = pyaudio.PyAudio()
sample_format = pyaudio.paInt16
channels = 1
fs = 512

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

def checkfun(txtname):
  for i in range(3):
    try:
      open(txtname, "r")
    except:
      time.sleep(0.5)
    else:
      break
  f = open(txtname, "r")
  countnum = 1024
  count = 0
  filename = txtname[:-4]
  b = []
    
  while (a := f.readline().split()) != []:
    b.append(int(round(float(a[1]))).to_bytes(10, byteorder='big'))
    count += 1
    if count >= countnum:
      wf = wave.open(f'{filename}.wav', 'wb')
      wf.setnchannels(channels)
      wf.setsampwidth(p.get_sample_size(sample_format))
      wf.setframerate(fs)
      wf.writeframes(b''.join(b))
      wf.close()
      break
  
  DATASET_PATH = filename + ".wav"
  data_dir = pathlib.Path(DATASET_PATH)
    
  x = data_dir
  x = tf.io.read_file(str(x))
  x, sample_rate = tf.audio.decode_wav(x, desired_channels=1)
  x = tf.squeeze(x, axis=-1)
  x = get_spectrogram(x)
  x = x[tf.newaxis,...]

  prediction = model.predict(x)[0]
  prediction = tf.cast(prediction, dtype=float)
  print(prediction)
  if tf.math.argmax(prediction) != 0 :
    print(f'{filename}漏水')
  else:
    print(f'{filename}OK')


while True:
  checklist = []
  for i in txtname:
    if os.path.isfile(i):
      check = thread.Thread(target=checkfun, name=f'{i}', args=(i,))
      check.start()
      checklist.append(check)
  
  checklist[-1].join()
  time.sleep(3)