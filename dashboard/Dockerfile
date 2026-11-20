FROM python:3.11-slim

WORKDIR /app
COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --create-home --uid 10001 verimind
COPY dashboard /app/dashboard
COPY dashboard/.streamlit /app/.streamlit
COPY results /app/results
COPY eval/results /app/eval/results
USER verimind
ENV PYTHONPATH=/app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '8501') + '/_stcore/health').read()"
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
