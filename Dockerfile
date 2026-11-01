FROM python:3.11-slim

# Run as a non-root user (required by Hugging Face Spaces, good practice everywhere)
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser . .
USER appuser

# Train the model at build time so the image is self-contained.
# Use --build-arg TRAIN_ARGS=--synthetic to build without internet access.
ARG TRAIN_ARGS=""
RUN python -m training.train ${TRAIN_ARGS}

# Hugging Face Spaces expects port 7860; other hosts set $PORT themselves.
ENV PORT=7860
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
