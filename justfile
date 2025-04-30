set dotenv-load
# This file is mostly for testing, but can be used as reference for what commands should look like

default:
  @just --list

up:
  docker compose up -d --build --force-recreate

up-miner:
  docker compose -f docker-compose.miner.yml up --build --force-recreate

down-miner:
  docker compose -f docker-compose.miner.yml down --remove-orphans

down:
  docker compose down --remove-orphans

update:
  git pull
  docker compose up -d --build --force-recreate

