FROM condaforge/miniforge3:latest
WORKDIR /work
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
COPY . .
SHELL ["conda", "run", "-n", "scvx-landing", "/bin/bash", "-c"]
ENV PYTHONPATH=/work/src
CMD ["conda", "run", "-n", "scvx-landing", "pytest", "-q"]
