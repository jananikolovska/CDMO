# Use Ubuntu as the base image
FROM ubuntu:20.04

# Set environment variable to prevent timezone selection prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, wget, and required Qt libraries for MiniZinc
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget \
    libqt5printsupport5 libqt5core5a libqt5gui5 libqt5widgets5 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and extract MiniZinc
RUN wget https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.4/MiniZincIDE-2.6.4-bundle-linux-x86_64.tgz && \
    tar -xvzf MiniZincIDE-2.6.4-bundle-linux-x86_64.tgz && \
    mv MiniZincIDE-2.6.4-bundle-linux-x86_64 /opt/minizinc && \
    rm MiniZincIDE-2.6.4-bundle-linux-x86_64.tgz

# Set MiniZinc path
ENV PATH="/opt/minizinc/bin:$PATH"

# Install Python packages
RUN pip3 install --no-cache-dir z3-solver numpy matplotlib pandas minizinc pulp

# Set the working directory
WORKDIR /app

# Set Python as the default entrypoint
CMD ["bash"]