# Use Ubuntu as the base image
FROM ubuntu:20.04

# Install Python and other required packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir z3-solver numpy matplotlib pandas

# Copy your Python script into the Docker image
COPY CDMO_SAT.py /app/CDMO_SAT.py
COPY check_solution.py /app/check_solution.py
COPY instances/ /app/instances/

# Set the working directory
WORKDIR /app

# Set Python as the default entrypoint
CMD ["bash"]