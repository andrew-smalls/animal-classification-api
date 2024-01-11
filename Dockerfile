FROM python:3.9.13-slim

WORKDIR /app
COPY . /app

# pip freeze > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Absolutely required for the app to work
COPY model.h5 /app/model.h5
COPY class_indices.json /app/class_indices.json
COPY static /app/static

EXPOSE 5000
EXPOSE 5001
EXPOSE 5002
EXPOSE 5003
EXPOSE 5004

ENV FLASK_APP=main

CMD ["flask", "run", "--host=0.0.0.0"]