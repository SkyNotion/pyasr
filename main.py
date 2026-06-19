import sounddevice as sd
import numpy as np
import threading
from scipy.io.wavfile import write
from readchar import readkey, key
from threading import Thread, Event
from queue import Queue
import uuid
import os
import httpx
import pyperclip
from time import time
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

USE_EXTERNAL_API = os.getenv("USE_EXTERNAL_API", "false").lower() == "true"
TRANSCRIPTION_BASE_URL = os.getenv("TRANSCRIPTION_BASE_URL", None)
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", None)
TRANSCRIPTION_API_KEY = os.getenv("TRANSCRIPTION_API_KEY", None)
CLEANER_BASE_URL = os.getenv("CLEANER_BASE_URL", None)
CLEANER_MODEL = os.getenv("CLEANER_MODEL", None)
CLEANER_API_KEY = os.getenv("CLEANER_API_KEY", None)

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

TERM_BOLD_WHITE = "\033[1;90m"
TERM_NORMAL = "\033[22m"
TERM_GREEN = "\033[32m"
TERM_BLUE = "\033[34m"
TERM_RESET = "\033[0m"

WHISPER_CPP = "whisper.cpp"
OPENAI_API = "openai.api"

http_client = httpx.Client(timeout=500000)

CLEANER_PROMPT = '''\
You are an expert prompt engineer. Your task is to take messy, raw ASR (speech-to-text) transcripts and convert them into clear text.
Rules:
1. Remove all conversational filler (um, ah, okay).
2. Fix obvious phonetic transcription errors.
3. Do not add any new technical requirements that were not implied in the original text.
4. Output ONLY the cleaned prompt. Do not include conversational replies like "Here is your text."'''

USE_CLEANER = os.getenv("USE_CLEANER", "false").lower() == "true"

def gen_short_uuid(prefix = ""):
    return f"{prefix}{str(uuid.uuid4()).split("-")[4]}"

def convert_wav_to_m4a(input_path):
    file_path = f"{AUDIO_ROOT}/{gen_short_uuid("rec-conv-")}.m4a"
    audio = AudioSegment.from_wav(input_path)
    audio.export(file_path, format="ipod", codec="aac")
    print(f"convert_wav_to_m4a - Converted to m4a - {os.path.basename(input_path)} -> {os.path.basename(file_path)}")
    return file_path

def llm_cleaner(text):
    if USE_EXTERNAL_API:
        response = http_client.post(
            url=f"{CLEANER_BASE_URL}/v1/chat/completions",
            json={
                "model": CLEANER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": CLEANER_PROMPT
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "stream": False
            },
            headers={
                "Authorization": f"Bearer {CLEANER_API_KEY}"
            },
            timeout=60
        )

        response.raise_for_status()
        response = response.json()
        return response["choices"][0]["message"]["content"]

    response = http_client.post(
        url="http://localhost:11434/api/generate",
        json={
            "model": "qwen3:0.6b",
            "system": CLEANER_PROMPT,
            "prompt": text,
            "stream": False,
            "think": True,
            "keep_alive": "1h",
            "options": {
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.8,
                "min_p": 0,
            }
        }
    )

    response = response.json()
    response = response["response"]
    response = [x.strip() for x in response.splitlines() if len(x) > 0]
    return "\n".join(response)

def inference(file_path, engine):
    if USE_EXTERNAL_API:
        response = http_client.post(
            url=f"{TRANSCRIPTION_BASE_URL}/v1/audio/transcriptions",
            data={
                "model": TRANSCRIPTION_MODEL,
                "response_format": "json"
            },
            files={
                "file": open(file_path, "rb")
            },
            headers={
                "Authorization": f"Bearer {TRANSCRIPTION_API_KEY}"
            },
            timeout=60
        )

        response.raise_for_status()
        response = response.json()
        return response["text"]

    if engine == WHISPER_CPP:
        response = http_client.post(
            url="http://127.0.0.1:9953/inference",
            data={
                #"temperature": "0.0",
                #"temperature_inc": "0.2",
                "response_format": "json"
            },
            files={
                "file": open(file_path, "rb")
            }
        )
    
        response = response.json()
        response = response["text"]
        response = [x.strip() for x in response.splitlines() if len(x) > 0]
        return "\n".join(response)
    elif engine == OPENAI_API:
        response = http_client.post(
            url="http://127.0.0.1:9953/v1/audio/transcriptions",
            data={
                "prompt": "language English<asr_text>",
                "model": "Qwen3-ASR",
                "response_format": "json",
                "language": "English",
                "temperature": "0.000001",
            },
            files={
                "file": open(file_path, "rb")
            }
        )

        response = response.json()
        response = response["text"].replace("language English<asr_text>", "")
        response = [x.strip() for x in response.splitlines() if len(x) > 0]
        return "\n".join(response)

def audio_callback(indata, frames, time, status):
    META["buffer"].append(indata.copy())

def run_audio_stream(file_path = None):
    if file_path is None:
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
    print(f"run_audio_stream - Transcribing...")

    start_time = time()

    if USE_EXTERNAL_API:
        file_path = convert_wav_to_m4a(file_path)

    response = inference(file_path, OPENAI_API)

    print("")
    print(f"{TERM_BLUE}Raw\n----------\n{response}{TERM_RESET}")
    print("")

    if USE_CLEANER:
        response = llm_cleaner(response)
    
        print("")
        print(f"{TERM_BLUE}Cleaned\n----------\n{response}{TERM_RESET}")
        print("")

    pyperclip.copy(response)

    print(f"run_audio_stream - Transcribed in {round(time() - start_time, 2)}s")
    META["buffer"].clear()
    META["feedback"].set()

def serve_audio():
    from http.server import SimpleHTTPRequestHandler
    import socketserver
    
    PORT = 11492
    try:
        handler = lambda *a, **kw: SimpleHTTPRequestHandler(*a, directory=AUDIO_ROOT, **kw)
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print(f"Serving audio files at port {PORT}")
            httpd.serve_forever()
    except Exception as e:
        print(f"serve_audio - Error - {e}")

def main():
    if not USE_EXTERNAL_API:
        Thread(target=serve_audio).start()
    print("main - Started")
    while True:
        if not META["active"]:
            print("main - Ready...")

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