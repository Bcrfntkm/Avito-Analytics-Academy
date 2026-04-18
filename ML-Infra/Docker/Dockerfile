FROM python:3.8
ENV FLASK_APP=server.py

LABEL version="1.0.0"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "server.py"]
