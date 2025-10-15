variable "REGISTRY" {
  default = "ghcr.io"
}

variable "IMAGE_NAME" {
  default = "allenai/rslearn_projects"
}

variable "TAGS" {
  default = ["latest"]
}

target "docker-metadata-action" {}

group "default" {
  targets = ["base", "full"]
}

target "base" {
  inherits = ["docker-metadata-action"]
  target = "base"
  tags = [for tag in TAGS : "${REGISTRY}/${IMAGE_NAME}:${tag}-base"]
  ssh = ["default"]
  args = {
    USE_SSH_KEYS_FROM_BUILD = "true"
  }
  cache-from = ["type=registry,ref=${REGISTRY}/${IMAGE_NAME}:buildcache-base"]
  cache-to = ["type=registry,ref=${REGISTRY}/${IMAGE_NAME}:buildcache-base,mode=max"]
}

target "full" {
  inherits = ["docker-metadata-action"]
  target = "full"
  tags = [for tag in TAGS : "${REGISTRY}/${IMAGE_NAME}:${tag}"]
  ssh = ["default"]
  args = {
    USE_SSH_KEYS_FROM_BUILD = "true"
  }
  cache-from = [
    "type=registry,ref=${REGISTRY}/${IMAGE_NAME}:buildcache-base",
    "type=registry,ref=${REGISTRY}/${IMAGE_NAME}:buildcache-full"
  ]
  cache-to = ["type=registry,ref=${REGISTRY}/${IMAGE_NAME}:buildcache-full,mode=max"]
}
