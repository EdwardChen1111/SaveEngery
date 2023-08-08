import matplotlib.pyplot as plt
import threading as thread
import tkinter as tk
import numpy as np
import pyaudio
import time
from tkinter import ttk
from tkinter import messagebox

fig, ax = plt.subplots()
now = ""
watertype = ""
soundnum = 0

def runStartGUI():
    global now
    now = "starting"
    ans = ""
    
    def startGUI():
        def startb():
            global watertype, soundnum
            nonlocal ans
            if now != "listen":
                ans, soundnum = "start", int(soundent.get())
                if watercombo.get() == '錄製正常水流聲':
                    watertype = 0
                else:
                    watertype = 1
            
        def exit():
            nonlocal ans
            ans = "exit"
            start.destroy()
        
        start = tk.Tk()
        start.title("startlist")
        start.geometry("240x130")
        watercombo = ttk.Combobox(start, width=20, height=2, values=['錄製正常水流聲','錄製漏水水流聲'])
        soundent = tk.Entry(start, width=20)
        startbtn = tk.Button(start, width=10, height=2, text="開始偵測", command=startb)
        exitbtn = tk.Button(start, width=10, height=2, text="結束程式", command=exit)
        watercombo.grid(row=0,columnspan=2)
        soundent.grid(row=1,column=0,columnspan=2)
        startbtn.grid(row=2,column=0,pady=20)
        exitbtn.grid(row=2,column=1,pady=20)
        
        start.mainloop()
    
    GUI = thread.Thread(target=startGUI)
    GUI.start()
    
    while True:
        time.sleep(0.1)
        if ans == "start":
            startListen = thread.Thread(target=startListening)
            startListen.start()
            now = "listen"
            ans = ""
        elif ans == "exit":
            now = "end"
            break

def startListening():
    global now, watertype, soundnum
    
    p = pyaudio.PyAudio()
    chunk = 44100
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=chunk)
    soundtemp = np.load("train.npz")
    soundarr, soundlab = soundtemp['x_train'], soundtemp['x_labels']
    soundarr = soundarr.tolist()
    soundlab = soundlab.tolist()
    
    for i in range(soundnum):
        if now != "end":
            frames = []
            for j in range(5):
                data = stream.read(chunk)
                frames += data
            for k in range(len(frames)):
                frames[k] /= 65535
            
            soundarr.append(frames)
            soundlab.append(watertype)
            print(len(frames), len(soundarr))
        else:
            break
    stream.stop_stream()
    stream.close()
    soundarr = np.array(soundarr)
    soundlab = np.array(soundlab)    
    
    np.savez_compressed("train.npz", x_train=soundarr, x_labels=soundlab)
    now = "starting"

if __name__ == '__main__':
    runStartGUI()