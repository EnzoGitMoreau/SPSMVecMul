# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables to avoid user interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install g++ and gcc
RUN apt-get update && \
    apt-get install -y g++ gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN  apt update && \
 apt install build-essential -y && apt-get install libopenblas-dev -y && apt-get install libboost-all-dev
# Set the working directory
WORKDIR /endpoint

# Copy your source code to the container (optional)
# COPY . .

# Default command to run when the container starts (optional)
# CMD ["bash"]