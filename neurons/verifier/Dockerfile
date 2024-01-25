FROM opentensorfdn/bittensor:6.6.0

COPY ../../ /app/targon
COPY app.py /app/app.py

WORKDIR /app


# install python, build essentials
RUN pip install -r targon/requirements.txt
RUN pip install uvicorn[standard]

EXPOSE 8000

#add your flag commands
CMD [ "uvicorn", "app:app", "--loop", "asyncio" ]