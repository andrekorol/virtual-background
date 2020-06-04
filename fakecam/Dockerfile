FROM python:3-buster

# ensure pip is up to date
RUN pip install --upgrade pip

# install OpenCV dependencies
RUN apt update && \
    apt install -y \
    `# OpenCV requirements` \
    libsm6 libxext6 libxrender-dev \
    `# OpenCV video opening requirements` \
    libv4l-dev

# install our Python requirements
WORKDIR /src
COPY requirements.txt /src/
RUN pip install --no-cache-dir -r /src/requirements.txt

# copy in the virtual background
COPY background.jpg /data/

# run our fake camera script (with unbuffered output for easier debug)
COPY fake.py /src/
ENTRYPOINT [ "python", "fake.py" ]