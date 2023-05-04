FROM python:3.10
ARG TARGETPLATFORM 
RUN echo $TARGETPLATFORM 

WORKDIR /app
COPY . /app 

RUN pip install -e . 
RUN pip install -e ".[test]"