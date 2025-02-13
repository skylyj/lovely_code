#!/usr/bin/env python
# coding: utf-8
import logging

import requests
from flask import Flask, abort, jsonify, request

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api_key = "sk-2896671c52df4c22b6372388d89d8da4"
base_url = "https://api.deepseek.com"
app = Flask(__name__)

SERVICE_API_KEY = "tolian"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    logger.debug("Received request: %s", request.json)

    # Check for API key in request headers
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning(
            "Unauthorized access attempt: Missing or invalid Authorization header"
        )
        abort(401, description="Unauthorized: Missing or invalid Authorization header")

    client_api_key = auth_header.split(" ")[1]
    if client_api_key != SERVICE_API_KEY:
        logger.warning("Unauthorized access attempt: Invalid API Key")
        abort(401, description="Unauthorized: Invalid API Key")

    headers = {
        "Content-Type": request.headers.get("Content-Type"),
        "Authorization": f"Bearer {api_key}",
    }
    try:
        response = requests.post(
            base_url + "/v1/chat/completions",
            headers=headers,
            json=request.json,
        )
        logger.debug("Response from DeepSeek API: %s", response.content)
        logger.debug("Status code: %d", response.status_code)
        logger.debug("Response headers: %s", response.headers.items())
    except requests.exceptions.RequestException as e:
        logger.error("Error communicating with DeepSeek API: %s", e)
        abort(500, description="Internal Server Error")

    return jsonify(response.json()), response.status_code


if __name__ == "__main__":
    try:
        logger.info("Starting Flask app with SSL context")
        app.run(
            host="0.0.0.0",
            port=443,
            ssl_context=(
                "/etc/letsencrypt/live/airocks.cn/fullchain.pem",
                "/etc/letsencrypt/live/airocks.cn/privkey.pem",
            ),
        )
    except Exception as e:
        logger.error("Failed to start Flask app: %s", e)




