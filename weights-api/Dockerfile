# Use official Python image as base
FROM python:3.12-slim

WORKDIR /app

# install system and Rust build dependencies
RUN apt-get update && apt-get install -y git curl build-essential pkg-config libssl-dev

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:/app/.venv/bin:${PATH}"

# dependency files first
COPY requirements.txt .

RUN pip install -r requirements.txt

# copy source packages
COPY . .

EXPOSE 8000
CMD ["python", "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

