# P0 integration runtime. Component owners add their pinned dependencies through
# EXTRA_REQUIREMENTS when their adapters are implemented; Airflow stays separate.
FROM python:3.11.11-slim-bookworm AS integration

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src
WORKDIR /workspace

COPY requirements/runtime.lock /tmp/runtime.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/runtime.lock \
    && useradd --create-home --uid 1000 --gid 0 app
COPY --chown=app:root . /workspace
ARG EXTRA_REQUIREMENTS=""
RUN if [ -n "$EXTRA_REQUIREMENTS" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends g++ libstdc++6 \
        && pip install --no-cache-dir --require-hashes -r "$EXTRA_REQUIREMENTS" \
        && apt-get purge -y --auto-remove g++ \
        && rm -rf /var/lib/apt/lists/*; \
    fi \
    && pip check \
    && mkdir -p /workspace/data /workspace/artifacts \
    && chown app:root /workspace/data /workspace/artifacts \
    && chmod g+rwX /workspace/data /workspace/artifacts

USER app
CMD ["python", "-m", "mlops_project.pipelines.worker", "--host", "0.0.0.0", "--port", "8100", "--project-root", "/workspace", "--config", "configs/project.yaml"]

# The API entrypoint is P3's handoff. Its absence causes a visible startup failure.
FROM integration AS api
USER root
RUN pip install --no-cache-dir --require-hashes -r requirements/api.lock && pip check
USER app
CMD ["python", "-m", "uvicorn", "mlops_project.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
