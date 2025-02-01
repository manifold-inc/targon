FROM python:3.9
WORKDIR /app

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./verifier.py .
COPY ./xgb_model.json .

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=15 CMD curl --silent --fail http://localhost/ > /dev/null || exit 1

ENTRYPOINT ["uvicorn", "verifier:app", "--port", "80", "--host", "0.0.0.0"]
