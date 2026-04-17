#Starts from the recommended slim Python base image to minimize size
FROM python:3.11-slim

#Sets the working directory inside the container
WORKDIR /app

#Copies the dependency file first to leverage Docker layer caching
COPY requirements.txt .

#Installs dependencies without storing pip cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

#Copies the rest of the application code
COPY app.py .

#Documents the port that Uvicorn will listen on
EXPOSE 8000

#Starts the FastAPI server listening on all network interfaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]