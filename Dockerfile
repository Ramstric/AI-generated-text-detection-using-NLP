# Use the official Python image from the Docker Hub
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY app .

EXPOSE 8080

ENTRYPOINT ["python"]

# Define the command to run the application
CMD ["app.py",  "--host=0.0.0.0"]

# Run docker build --tag northamerica-northeast1-docker.pkg.dev/ai-detection-demo-452904/ai-detection-api/api-demo .
# gcloud auth login
# gcloud auth print-access-token account@account.com
# docker login -u oauth2accesstoken -p "token-string" https://northamerica-northeast1-docker.pkg.dev
# docker push northamerica-northeast1-docker.pkg.dev/ai-detection-demo-452904/ai-detection-api/api-demo
# Test with curl -X POST "https://api-demo-2-182443968648.northamerica-northeast1.run.app/api/predict" -d text="This is a test"