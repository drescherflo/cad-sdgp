FROM python:3.10

# Set Working directory
WORKDIR /config_generation

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy generator script
COPY 6_dof_config_generator.py .

# Set generator as entrypoint
ENTRYPOINT ["python", "6_dof_config_generator.py"]
