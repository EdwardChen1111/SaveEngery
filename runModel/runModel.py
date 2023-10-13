import tensorflow as tf
import pygsheets
import pathlib
import pyaudio
import time
import wave

gc = pygsheets.authorize(service_file='Google python.json')

sht = gc.open_by_url(
  'https://docs.google.com/spreadsheets/d/10QLJQ637W10XsH1vfdtxty7J3zbZomKTtStDnKmp7ho/edit?usp=drivesdk'
)

audiolist = {'sample_format':pyaudio.paInt16, 'channels':1, 'fs':10000, 'data_dir':pathlib.Path("temp.wav")}
model = tf.keras.models.load_model('saved_model')
nownum = len(sht[0].get_row(3))
p = pyaudio.PyAudio()

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def loadwave(data_dir):
  x = data_dir
  x = tf.io.read_file(str(x))
  x, sample_rate = tf.audio.decode_wav(x, desired_channels=1)
  x = tf.squeeze(x, axis=-1)
  x = get_spectrogram(x)
  x = x[tf.newaxis,...]
  return x

def savewave(wks, cellpos):
  cel = wks.cell(f'C{cellpos}')
  templist = [int(round(float(i))).to_bytes(10, byteorder='big') for i in cel.value.split("/")[:-1]]

  wf = wave.open('temp.wav', 'wb')
  wf.setnchannels(audiolist['channels'])
  wf.setsampwidth(p.get_sample_size(audiolist['sample_format']))
  wf.setframerate(audiolist['fs'])
  wf.writeframes(b''.join(templist))
  wf.close()
  
def predict(x):
  prediction = model.predict(x, verbose=0)[0]
  prediction = tf.cast(prediction, dtype=float)
  if tf.math.argmax(prediction) != 2 :
    print('漏水')
  else:
    print('無漏水')
  print()

def checkfunction(data_dir):
  global nownum, sht
  
  wks = sht[0]
  while (cel := wks.cell(f'C{nownum}').value) != "":
    nownum += 1
  nownum -= 1
  
  if nownum != 0:
    savewave(wks, nownum)
    predict(loadwave(data_dir))
    
if __name__ == '__main__':
  while True:
    checkfunction(audiolist['data_dir'])
    time.sleep(2)