FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && addgroup --system paper \
    && adduser --system --ingroup paper --home /app paper \
    && mkdir -p /data \
    && chown -R paper:paper /app /data

COPY --chown=paper:paper app ./app
COPY --chown=paper:paper pyproject.toml README_RU.md ./

USER paper

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health', timeout=3)"

CMD ["python", "-m", "app"]

