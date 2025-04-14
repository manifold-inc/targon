FROM python:3.12

WORKDIR /app


# Make sure we arent missing anything
RUN apt-get -y update; apt-get -y install curl
COPY gpu_remote_policy.json .
COPY switch_remote_policy.json .
COPY requirements.txt . 
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .

HEALTHCHECK --interval=1s --timeout=5s --retries=15 CMD curl --silent --fail http://0.0.0.0 > /dev/null || exit 1

RUN export PYTHONUNBUFFERED=1
ENTRYPOINT ["python",  "-m", "uvicorn", "main:app", "--port", "80", "--host", "0.0.0.0"]
