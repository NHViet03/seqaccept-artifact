# Shared base: Python of the IDoFT commits' era (2017-2020), git, pytest + pytest-randomly (Def. 4: seeded random order).
FROM python:3.8-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "pytest==7.4.4" "pytest-randomly==3.12.0"
