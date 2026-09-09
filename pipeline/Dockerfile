# Refresh job container — the image the Phase 8 Lambda runs (8-01) and that
# runs standalone with a mounted output dir. Writes only to --out.
FROM python:3.12-slim

WORKDIR /job
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "hedging_workbench.refresh"]
CMD ["--out", "/out"]
