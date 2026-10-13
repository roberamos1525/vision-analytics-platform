# NOTE: written for portability but NOT built locally (Docker is not installed on the
# development machine). See README "Deployment".

FROM python:3.11-slim AS build
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r requirements.lock \
 && /opt/venv/bin/pip install --no-deps .

FROM python:3.11-slim
# libgomp is needed by the OpenCV wheel; ffmpeg is bundled inside opencv-python-headless.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 vap \
 && mkdir -p /var/lib/vap /media/vap \
 && chown vap:vap /var/lib/vap
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    VAP_SERVER__HOST=0.0.0.0 \
    VAP_SERVER__PORT=8080 \
    VAP_STORAGE__STATE_DIR=/var/lib/vap \
    VAP_MEDIA__ROOTS='["/media/vap"]'
USER vap
WORKDIR /home/vap
VOLUME ["/var/lib/vap"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=2).status == 200 else 1)"
CMD ["vap", "serve"]
