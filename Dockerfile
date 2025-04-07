# use the official PyTorch image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# set the working directory
WORKDIR /chat-bot
COPY . /chat-bot

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyter ipykernel

# copy the dependencies file to the working directory
COPY notebooks/ chat-bot/notebooks/
COPY src/ chat-bot/src/
COPY configs/ /chat-bot/configs/

ENV PYTHONPATH="${PYTHONPATH}:/chat-bot"

# Expose notebook port
EXPOSE 8888

# start the notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--notebook-dir=chat-bot/notebooks"]
