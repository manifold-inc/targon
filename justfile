set dotenv-load
# This file is mostly for testing, but can be used as reference for what commands should look like

default:
  @just --list

up:
  docker compose up -d --build --force-recreate targon nvidia-attest

up-miner:
  docker compose -f docker-compose.miner.yml up -d --build --force-recreate
  docker compose -f docker-compose.miner.yml logs -f

down-miner:
  docker compose -f docker-compose.miner.yml down --remove-orphans

down:
  docker compose down --remove-orphans

update:
  git pull
  docker compose pull
  docker compose up -d --build --force-recreate targon nvidia-attest

install-cli:
  cd targon && go install ./cmd/targon-cli
