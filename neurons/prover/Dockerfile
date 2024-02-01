FROM opentensorfdn/bittensor:6.6.0

# COPY ../../. /app/targon
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
WORKDIR /app

RUN git clone https://github.com/manifold-inc/targon.git /app/targon

# install python, build essentials

RUN pip install -e /app/targon
RUN pip install uvicorn[standard]

EXPOSE 8091

RUN chmod +x /app/entrypoint.sh

CMD ["./app/entrypoint.sh"]
#add your flag commands