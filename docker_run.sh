#!/bin/bash
docker run --rm --runtime=nvidia --gpus all -p 8888:8888 \
    -v $(pwd)/data:/chat-bot/data \
    -v $(pwd)/models:/chat-bot/models \
    -v $(pwd)/audios:/chat-bot/audios \
    -v $(pwd)/audios:/chat-bot/output \
    -it chat-bot
    