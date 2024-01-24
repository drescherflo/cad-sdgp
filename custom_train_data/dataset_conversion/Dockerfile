FROM python:3.10

# Set Working directory
WORKDIR /dataset_converter

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy converter script and plugins
COPY dataset_converter.py .
COPY converter_plugins .

# Set generator as entrypoint
ENTRYPOINT ["python", "dataset_converter.py"]
