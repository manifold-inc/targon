FROM python:3.12.2
WORKDIR /app

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./verifier.py .
COPY ./shared.py .

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --start-interval=30s --retries=15 CMD curl --silent --fail http://localhost/ > /dev/null || exit 1

ENTRYPOINT ["uvicorn", "verifier:app", "--port", "80", "--host", "0.0.0.0"]
