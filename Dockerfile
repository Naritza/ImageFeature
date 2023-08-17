FROM python:3.10.6
 
WORKDIR /app


COPY requirements.txt .


RUN pip3 install --no-cache-dir -r requirements.txt


RUN apt-get update && apt-get install -y libgl1-mesa-glx


COPY app /app


EXPOSE 8080


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]