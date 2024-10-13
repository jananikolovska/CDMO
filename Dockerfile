# Use Ubuntu as the base image
FROM ubuntu:20.04

# Install Python and other required packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir z3-solver numpy matplotlib jupyter

# Expose the port that Jupyter runs on
EXPOSE 8888

# Run Jupyter Notebook without a token
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]


# Set Python as the default entrypoint
#CMD ["python3"]
