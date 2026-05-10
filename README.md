# pyasr

Use [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for Automatic Speech Recognition (ASR).

Just press space to talk (DO NOT HOLD), And press space to transcribe when done.

## Requirements

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) server must be running locally at port `9953`.

[uv](https://github.com/astral-sh/uv) must be installed.

## Running

```bash
git clone https://github.com/SkyNotion/pyasr.git
cd pyasr
uv sync
uv run main.py

```