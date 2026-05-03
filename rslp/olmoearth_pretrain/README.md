This module contains:

- Script to compute and save OlmoEarth embeddings for a target dataset. This is mainly
  for AEF evaluation. See `get_embeddings.py`.
- Instructions below for building a Docker image with rslearn, rslearn_projects, and olmoearth_pretrain.

## olmoearth_pretrain.Dockerfile

Create a copy of `rslearn_projects` repository with subfolders `docker_build/rslearn`
(containing https://github.com/allenai/rslearn) and `docker_build/olmoearth_pretrain`
(containing https://github.com/allenai/olmoearth_pretrain). Then run:

    DOCKER_BUILDKIT=1 docker build -t rslpomp -f olmoearth_pretrain.Dockerfile .
    beaker image create --name rslpomp rslpomp

The Dockerfile requires BuildKit (for `--mount=type=cache`). If `docker buildx` is not
installed, install it first:

    mkdir -p ~/.docker/cli-plugins
    curl -Lo ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/v0.32.1/buildx-v0.32.1.linux-amd64
    chmod +x ~/.docker/cli-plugins/docker-buildx
