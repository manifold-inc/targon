FROM python:3.12

WORKDIR /app


# Make sure we arent missing anything
COPY requirements.txt . 
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .

RUN apt-get -y update; apt-get -y install curl
HEALTHCHECK --interval=1s --timeout=5s --retries=15 CMD curl --silent --fail http://0.0.0.0 > /dev/null || exit 1

ENTRYPOINT ["uvicorn", "main:app", "--port", "80", "--host", "0.0.0.0"]
