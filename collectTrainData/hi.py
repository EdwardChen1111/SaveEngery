import wave
import pyaudio

f = open("Bpoint.txt", "r")
countnum = 1024
count = 0
filenum = 1
b = []

p = pyaudio.PyAudio()
sample_format = pyaudio.paInt16
channels = 1
fs = 5120

while (a := f.readline().split()) != []:
    b.append(int(round(float(a[1]))).to_bytes(10, byteorder='big'))
    count += 1
    if count >= countnum:
        wf = wave.open(f'{filenum}.wav', 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(b))
        wf.close()
        count = 0
        filenum += 1
        b = []
        print("///////////////////////")