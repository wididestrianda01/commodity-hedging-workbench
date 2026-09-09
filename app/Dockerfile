FROM python:3.12-slim

WORKDIR /w
# Editable install: FROZEN_DIR resolves against /w/src, so /w/data is found
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY app ./app
RUN pip install --no-cache-dir -e ".[app]"

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
