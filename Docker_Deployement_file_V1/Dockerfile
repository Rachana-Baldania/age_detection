FROM python:3.9

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install pillow
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt


# Run the application on port 5000
EXPOSE 5000
CMD ["python", "app.py"]
