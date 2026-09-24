# Hugging Face Space (sdk: docker) and anywhere else you want a container.
# CPU-only torch wheel keeps the image ~1.5 GB instead of ~6 GB.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=8001 \
    JEB_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8001
CMD ["python", "server.py"]
