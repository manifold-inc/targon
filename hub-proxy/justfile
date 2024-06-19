
GREEN  := "\\u001b[32m"
RESET  := "\\u001b[0m\\n"
CHECK  := "\\xE2\\x9C\\x94"

set shell := ["bash", "-uc"]

default:
  @just --list

build opts = "":
  docker compose build {{opts}}
  @printf " {{GREEN}}{{CHECK}} Successfully built! {{CHECK}} {{RESET}}"

pull:
  @git pull

up extra='': build
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --force-recreate {{extra}}
  @printf " {{GREEN}}{{CHECK}} Images Started {{CHECK}} {{RESET}}"

prod: build
  docker compose up -d
  @printf " {{GREEN}}{{CHECK}} Images Started {{CHECK}} {{RESET}}"

upgrade: pull build
  docker compose up -d proxy uid-cacher
  @printf " {{GREEN}}{{CHECK}} Images Started {{CHECK}} {{RESET}}"

down:
  @docker compose down
