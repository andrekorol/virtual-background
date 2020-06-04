# Base image with  TensorFlow requirements
FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Install Node.js
RUN apt update && apt install -y curl make build-essential \
    && curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt update && apt install -y nodejs \
    && mkdir /.npm \
    && chmod 777 /.npm

# Ensure we can get enough GPU memory
# Unfortunately tfjs-node-gpu exposes no gpu configuration
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Install npm dependencies
WORKDIR /src
COPY package.json /src/
RUN npm install

# Set our app as the entrypoint
COPY app.js /src/
ENTRYPOINT [ "node", "/src/app.js" ]