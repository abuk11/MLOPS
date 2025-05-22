FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY conda.yaml /src/conda.yaml
WORKDIR /src
RUN conda env create -f conda.yaml && conda clean -afy

COPY / /src

RUN cp /src/entrypoint.sh /entrypoint.sh && \
    sed -i 's/\r$//' /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENV PATH=/opt/conda/envs/cifar-env/bin:$PATH

EXPOSE 4272 8888

ENTRYPOINT ["bash", "/entrypoint.sh"]
