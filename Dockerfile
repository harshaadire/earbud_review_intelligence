FROM python:3.11-slim

WORKDIR /app

#installing system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#install the python dependencies also here \
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Pre-download the nltk data at first so that we have the headace of fetching data everytime
RUN python -m nltk.downloader punkt punkt_tab

# copy the rest of the project
COPY . .

#EXPOSE the both ports here
EXPOSE 8000 8501

#Start the script runs both service together
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]