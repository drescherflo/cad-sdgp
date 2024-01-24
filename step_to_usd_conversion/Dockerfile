FROM continuumio/miniconda3

# Set working directory
WORKDIR step_to_obj_converter

# Create environment
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Set the following RUN commands to use the environment
SHELL ["conda", "run", "-n", "step-to-obj", "/bin/bash", "-c"]

# Copy converter script
COPY step_to_obj.py .

# Set converter script as entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "step-to-obj", "python", "step_to_obj.py"]
