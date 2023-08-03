import matplotlib.pyplot as plt
import threading as thread
import tkinter as tk
import numpy as np
import pyaudio
import time
import wave

fig, ax = plt.subplots()
now = ""

def runStartGUI():
    global now
    now = "starting"
    ans = ""
    
    def startGUI():
        def btn(msg):
            nonlocal ans
            ans = msg
                
        def exit():
            btn("exit")
            start.destroy()
        
        start = tk.Tk()
        start.title("startlist")
        start.geometry("360x60")
        startbtn = tk.Button(start, width=10, height=2, text="開始偵測", command=lambda:btn("start")).grid(row=0,column=0)
        endbtn = tk.Button(start, width=10, height=2, text="結束偵測", command=lambda:btn("end")).grid(row=0,column=1)
        exitbtn = tk.Button(start, width=10, height=2, text="結束程式", command=exit).grid(row=0,column=2)
        start.mainloop()
    
    GUI = thread.Thread(target=startGUI)
    GUI.start()
    
    while ans != "exit":
        time.sleep(0.1)
        if ans == "start":
            startListen = thread.Thread(target=startListening)
            startListen.start()
            now = "listen"
            ans = ""
        elif ans == "end":
            now = "end"
            ans = ""
            
    now = "end"
    return 0

def startListening():
    global now
    p = pyaudio.PyAudio()
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    filename = "oxxostudio.wav"
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []
    while now != "end":
        data = stream.read(chunk)
        frames.append(data)
        if len(frames) > 3400:
            frames.pop(0)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def visualize(path):
    raw = wave.open(path)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
    f_rate = raw.getframerate()
    time = np.linspace(0, len(signal)/f_rate, num = len(signal))

    ax.plot(time, signal)
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.show()

if __name__ == '__main__':
    runStartGUI()
    visualize("oxxostudio.wav")