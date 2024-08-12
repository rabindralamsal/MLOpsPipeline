FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip install -r requirements.txt

CMD ["uvicorn", "application:application", "--host", "0.0.0.0", "--port", "9191"]