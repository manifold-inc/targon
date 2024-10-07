FROM python:3.9
WORKDIR /app

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./verifier.py .


ENTRYPOINT ["uvicorn", "verifier:app", "--port", "80", "--host", "0.0.0.0"]
