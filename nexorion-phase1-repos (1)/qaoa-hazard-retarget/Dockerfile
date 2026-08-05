FROM condaforge/miniforge3:latest
WORKDIR /work
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
COPY . .
ENV PYTHONPATH=/work/src
CMD ["conda", "run", "-n", "qaoa-hazard-retarget", "pytest", "-q"]
