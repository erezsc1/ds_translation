FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
FROM huggingface/transformers-pytorch-gpu
FROM nvidia/cuda
WORKDIR /usr/src
COPY . .
RUN ls -l
RUN apt update && apt install python3.7 python3-pip -y
RUN pip3 install -r requirements.txt
EXPOSE 80
ENV LANG C.UTF-8
ENV LANGUAGE C:en
ENV LC_ALL C.UTF-8

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]