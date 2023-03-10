FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
COPY app/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY app /app

CMD ["python", "server.py"]