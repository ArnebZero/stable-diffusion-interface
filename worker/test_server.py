# pylint: disable=W0603
import secrets

import numpy as np
from flask import jsonify, request, Flask
from omegaconf import OmegaConf

from logger import get_logger

ENGLISH_ALPABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
WHITESPACE = " "

ALPHABET = ENGLISH_ALPABET + WHITESPACE
ALPHABET_LIST = list(ALPHABET)

app = Flask(__name__)
TOKEN = None
CONF = None
LOGGER = None


def get_word(min_len: int, max_len: int):
    assert min_len > 0, min_len
    assert min_len < max_len, (min_len, max_len)

    length = np.random.randint(min_len, max_len)

    return "".join(np.random.choice(ALPHABET_LIST, length)).lower()


def get_task(size: int, conf):
    texts = []
    ids = []

    for _ in range(size):
        texts.append(get_word(conf.min_len, conf.max_len))
        ids.append(secrets.token_hex(8))

    query = {
        "result": size,
        "data": [
            {"id": item_id, "text": text}
            for item_id, text in zip(ids, texts)
        ],
    }

    return query

def get_nothing_task():
    query = {
        "result": 0,
    }

    return query


def get_random_task(conf: OmegaConf):
    task_types = ["text", "nothing"]

    task = np.random.choice(task_types)

    if task == "text":
        size = np.random.randint(conf.text_task.min_size, conf.text_task.max_size)
        return get_task(size, conf=conf.text_task)

    return get_nothing_task()


@app.route("/api/v1/stage_sd/get_task", methods=["POST"])
def send_query():
    assert request.json["token"] == TOKEN

    task = get_random_task(CONF)

    info = f"Sending task with len {task['result']}"

    LOGGER.info(info)

    return jsonify(task)


@app.route("/api/v1/stage_sd/result", methods=["POST"])
def recieve_answer():
    assert request.json["token"] == TOKEN

    for item in request.json["data"]:
        assert len(item["images"]["0"]) == 512 * 512 * 3
        assert len(item["images"]["1"]) == 512 * 512 * 3
        assert len(item["images"]["2"]) == 512 * 512 * 3

    info = f"Got task with len: {request.json['result']}"

    LOGGER.info(info)

    return ("", 200)


def main():
    conf = OmegaConf.load("server_config.yaml")

    global TOKEN
    TOKEN = conf.token

    global CONF
    CONF = conf

    global LOGGER
    LOGGER = get_logger("test_server_logger")


if __name__ == "__main__":
    main()
    app.run(debug=True)
