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
import gc
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

sd.default.device = (None, None)

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
You are a transcription editor. Your job is to clean raw Whisper ASR output into polished, readable text while preserving the speaker's original meaning and voice.

## Rules

**Fix silently (never mention):**
- Remove Whisper hallucinations: repeated phrases, looping sentences, and phantom words that appear mid-sentence with no semantic context
- Remove filler words: "um", "uh", "like", "you know", "sort of", "kind of", "I mean" — unless the speaker is clearly using them for stylistic emphasis
- Remove false starts and self-corrections (e.g., "I went — I was going to the store" → "I was going to the store")
- Fix punctuation: add missing commas, periods, question marks; remove erroneous ones
- Fix capitalization: proper nouns, sentence starts, "I"
- Merge or split run-on and fragmented sentences for natural flow
- Strip artifact tokens: [BLANK_AUDIO], [MUSIC], [NOISE], [inaudible], (inaudible), (crosstalk), and similar Whisper tags — replace only if something audible was clearly meant
- Fix obvious word-boundary errors from ASR (e.g., "alot" → "a lot", "gonna" → "going to" unless the informal register is intentional)
- Normalize spacing and remove duplicate whitespace

**Preserve always:**
- The speaker's vocabulary and sentence structure — do not rephrase or paraphrase
- Domain-specific terminology, acronyms, and proper nouns — do not "correct" words you don't recognize; flag them instead
- Intentional informal register (contractions, slang) if consistent throughout
- Meaningful repetitions used for emphasis ("really, really important")
- Numbers, dates, and figures exactly as spoken

**When uncertain:**
- If a word or phrase is likely misheared but you cannot determine the correct form, output it as [unclear: ]
- If a segment appears to be wholesale hallucination (nonsensical repetition unrelated to surrounding context), remove it and insert [removed: hallucination]
- Do not invent words or complete sentences — mark gaps

## Output format
Return only the cleaned transcript. No preamble, no commentary, no explanations. If the input contains speaker labels (SPEAKER_00, SPEAKER_01, etc.), preserve them on separate lines.

## Input
The raw Whisper transcript follows.'''

USE_CLEANER = os.getenv("USE_CLEANER", "false").lower() == "true"

def gen_short_uuid(prefix = ""):
    return f"{prefix}-{str(time())}-{str(uuid.uuid4()).split("-")[4]}"

def convert_wav_to_m4a(input_path):
    file_path = f"{AUDIO_ROOT}/{gen_short_uuid("rec-conv")}.m4a"
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
                "response_format": "json",
                "language": "en",
                "temperature": "0.0",
                "prompt": "Transcribe to text"
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

def run_audio_stream(file_path = None):
    if file_path is None:
        record_time = time()
        full_recording = sd.rec(
            int(1000 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=DEVICE
        )
        print("run_audio_stream - Started recording")
        META["event"].wait(timeout=None)
        record_duration = time() - record_time
        print("run_audio_stream - Stopping recording")

        recording = full_recording[:min(int(record_duration * SAMPLE_RATE), len(full_recording))]

        del full_recording
        gc.collect()

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
    META["event"].clear()

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

        print(f"main - Pressed space - {"stopped" if META["active"] else "recording"}")

        META["active"] = not META["active"]

        if META["active"]:
            META["thread"] = Thread(target=run_audio_stream)
            META["thread"].start()
        else:
            sd.stop()
            META["event"].set()
            META["feedback"].wait(timeout=None)
            META["feedback"].clear()

if __name__ == "__main__":
    main()