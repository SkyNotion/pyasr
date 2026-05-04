import sounddevice as sd
import numpy as np
import threading
from scipy.io.wavfile import write
from readchar import readkey, key
from threading import Thread, Event
from queue import Queue
import uuid
import os

AUDIO_ROOT = os.path.join(os.getcwd(), "audio")

if not os.path.exists(AUDIO_ROOT):
    os.mkdir(AUDIO_ROOT)

SAMPLE_RATE = 48000
CHANNELS = 2
DEVICE = 0
BLOCK_SIZE  = 2048
META = {
    "active": False,
    "buffer": [],
    "event": Event(),
    "feedback": Event(),
    "thread": None,
    "queue": Queue()
}

def gen_short_uuid(prefix = ""):
    return f"{prefix}{str(uuid.uuid4()).split("-")[4]}"

def audio_callback(indata, frames, time, status):
    META["buffer"].append(indata.copy())

def run_audio_stream():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, latency="high",
                blocksize=BLOCK_SIZE, callback=audio_callback, device=DEVICE):
        print("run_audio_stream - Started recording")
        META["event"].wait(timeout=None)
        print("run_audio_stream - Stopping recording")
        META["event"].clear()

    recording = np.concatenate(META["buffer"], axis=0)
    file_path = f"{AUDIO_ROOT}/{gen_short_uuid("rec")}.wav"
    print(f"run_audio_stream - Saving file to {file_path}")
    write(file_path, SAMPLE_RATE, recording)
    print(f"run_audio_stream - Saved file")
    META["feedback"].set()

def main():
    while True:
        k = readkey()
        if k != key.SPACE:
            continue

        print(f"main - Pressed space - {"recording" if META["active"] else "stopped"}")

        META["active"] = not META["active"]

        if META["active"]:
            META["thread"] = Thread(target=run_audio_stream)
            META["thread"].start()
        else:
            META["event"].set()
            META["feedback"].wait(timeout=None)
            META["feedback"].clear()

if __name__ == "__main__":
    main()