# Mongo Wrapper Service

The Mongo Wrapper Service provides a RESTful HTTP API to access auction results
from the MongoDB database used by the Targon validator and miner services.

## Overview

This service acts as a secure middleware layer between external applications and
the MongoDB database, providing:

- RESTful API endpoint for auction results
- Health checks and monitoring
- Secure access to auction data only

## Features

### Auction Results Access

- Retrieve auction results with pagination
- Support for query parameters (limit)
- Timestamp and block information included

## API Endpoints

### Auction Results

- `GET /api/v1/auction-results` - Get auction results from the latest intervals

## Configuration

The service is configured via environment variables:

### Environment Variables

- `MONGO_USERNAME` - MongoDB username
- `MONGO_PASSWORD` - MongoDB password
- `ENABLE_MONGO_WRAPPER` - `true` or `false`
- `MONGO_WRAPPER_HOST` - your hostname

> NOTE: you must also set your letsencrypt email in traefik/traefik.https.toml.
> Just copy the template file and add your email to configure.

### Pagination

- `limit` - Number of auction results to return (default: 10)

### Examples

```
GET /api/v1/auction-results
GET /api/v1/auction-results?limit=5
```

## Response Format

All endpoints return JSON responses with the following structure:

### Success Response

```json
{
  "data": [
    {
      "timestamp": 1234567890,
      "block": 12345,
      "auction_results": {
        "auction_type": [
          {
            "price": 50,
            "uid": "123",
            "gpus": 4,
            "payout": 2.0,
            "diluted": false
          }
        ]
      }
    }
  ]
}
```
