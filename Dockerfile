FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install base binary packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y --no-install-recommends \
    install tzdata gcc python3-dev python3-setuptools curl lsb-release python3-venv python3-numpy gnupg2 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Install gcloud
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-jessie main" |  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg |  apt-key add -
RUN apt-get update && apt-get install --no-install-recommends -y google-cloud-sdk python3-crcmod kubectl && rm -rf /var/lib/apt/lists/*

#setup venv
RUN python3.6 -m venv /env
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ENV PYTHONPATH="${PYTHONPATH}:/app"
RUN pip install --no-cache-dir -U pip
ENV SM_FRAMEWORK = 'tf.keras'

# Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
