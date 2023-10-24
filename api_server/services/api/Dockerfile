FROM python:3.11.1-slim

COPY ~/.bittensor/wallets /app/wallets

COPY . /app

WORKDIR /app


# install python, build essentials
RUN pip install -r requirements.txt
RUN pip install uvicorn[standard]

EXPOSE 8000

#add your flag commands
CMD [ "python", "server.prod.py" ]