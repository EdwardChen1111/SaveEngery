import librosa
import wave
import soundfile as sf
# to install librosa package
# > conda install -c conda-forge librosa 
filenum = 1

while True:
    try:
        wf = wave.open(f'{filenum}.wav', 'rb')
        wf.close()
    except:
        break
    else:
        y, sr = librosa.load(f'{filenum}.wav', sr=5120)
        y_10k = librosa.resample(y,sr,10000)

        sf.write(f'{filenum}.wav', y_10k, 10000)
        filenum += 1
        print(filenum)


