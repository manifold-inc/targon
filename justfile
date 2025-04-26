set dotenv-load
# This file is mostly for testing, but can be used as reference for what commands should look like

default:
  @just --list

up:
  docker compose up -d --build --force-recreate

down:
  docker compose down --remove-orphans

update:
  git pull
  docker compose up -d --build --force-recreate

