FROM opentensorfdn/bittensor:6.6.0

COPY ../../ /app/targon
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
WORKDIR /app


# install python, build essentials
RUN pip install -e targon
RUN pip install uvicorn[standard]

EXPOSE 8000

RUN chmod +x /app/entrypoint.sh
#add your flag commands