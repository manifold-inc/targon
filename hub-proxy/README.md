# Hub Proxy setup

This is the backend for a targon hub instance. Includes a traefik instance for
scaling, uid-cacher for getting top miners and inserting them in a redis cache,
and the proxy itself. Note this does not need to run on the same server as the
validator itself.

## 1. Copy sample.env to .env

## 2. Assign subdomain to validator

Go to your DNS provider and assign a domain (or subdomain) to the server running
the proxy. If you dont want to use a domain and proxy via HTTP, just remove the
traefik https config. Regardless, put the domain/ip in the .env as `PROXY_URL`

If you assign a doamin, make sure to set your email in `traefik/traefik.toml`
under `[certificatesResolvers.letsencrypt.acme]`. This is used to generate your
ssl certificate.

## 3. Get your PUBLIC_KEY and PRIVATE_KEY from your wallet

Run `python scripts/setup.py --wallet [your wallet name]` and copy the ss58,
public, and private keys to the .env file

## 4. Copy your HUB_SECRET_KEY from the FE

## 5. Start docker compose

Run `docker compose build` and `docker compose up -d`

Make sure everything starts up correctly via `docker compose logs -f`
