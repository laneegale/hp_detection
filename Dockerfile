FROM nvcr.io/nvidia/pytorch:26.06-py3

# system packages
RUN apt-get update && apt-get install -y curl ...

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# copy dependency metadata first
COPY pyproject.toml uv.lock ./
COPY trident ./trident

# or copy the full repo if local path deps require more files
COPY . .

RUN uv sync --frozen

CMD ["./.venv/bin/python", "fine-tune.py", "/mnt/Z/cuhk_data/HPACG/batch2/data", "./runs0709", "--eval-data-dir", "/mnt/Z/cuhk_data/HPACG/batch1/data", "--batch-size", "8", "--num-workers", "4", "--epochs", "20", "--model", "conch_v15"]