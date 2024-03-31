# Use the official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install pillow
RUN pip install --no-cache-dir -r requirements.txt

# Run the application on port 5000
EXPOSE 5000
CMD ["python", "app.py"]


##Also Pull from Docker : https://hub.docker.com/r/rachanasimform/age_detection/tags by using command line docker pull rachanasimform/age_detection:latest