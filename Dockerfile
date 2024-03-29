FROM python:3.6-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg -y 

COPY requirements.txt .
RUN python3 -m venv ./venv && ./venv/bin/python3 -m pip install --upgrade pip && ./venv/bin/pip install -r requirements.txt

COPY . .

ENTRYPOINT ["./venv/bin/python", "tts.py"]
