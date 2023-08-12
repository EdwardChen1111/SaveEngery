import matplotlib.pyplot as plt
import threading as thread
import tkinter as tk
import numpy as np
import pyaudio
import time
import wave
import os
from tkinter import ttk
from tkinter import messagebox

fig, ax = plt.subplots()
now = ""
watertype = ""
soundnum = 0
dirc = os.getcwd()
if not ('train' in os.listdir()):
    os.mkdir('train')
os.chdir(dirc + '/train')

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
    
    if not (str(watertype) in os.listdir()):
        os.mkdir(str(watertype))
    filenum = len(os.listdir(str(watertype)))
    os.chdir(dirc + '/train/' + str(watertype))
    
    
    p = pyaudio.PyAudio()
    chunk = 16000
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 16000
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=chunk)
    
    for i in range(soundnum):
        if now != "end":
            filenum += 1
            frames = []
            for j in range(1):
                data = stream.read(chunk)
                frames.append(data)
            
            wf = wave.open(f'{filenum}.wav', 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))
            wf.close()
        else:
            break
    
    stream.stop_stream()
    stream.close()
    os.chdir(dirc + '/train')
    
    now = "starting"

if __name__ == '__main__':
    runStartGUI()