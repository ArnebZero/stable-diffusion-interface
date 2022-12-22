# pylint: disable=W0703
import time
import traceback
from logging import Logger
from typing import Dict

import requests
from omegaconf import OmegaConf

from logger import get_logger


def get_task(conf: OmegaConf) -> Dict:
    query = {"token": conf.token}

    response = requests.post(conf.url_get, json=query, timeout=conf.timeout)
    response.raise_for_status()

    if response.json()["result"] == 0:
        if "retry_after_seconds" in response.json():
            waiting_time = response.json()["retry_after_seconds"]
        else:
            waiting_time = conf.retry_after_seconds
        time.sleep(waiting_time)

        output_query = {}

    else:
        output_query = response.json()

    return output_query


def model_calculation(query: Dict, conf: OmegaConf) -> Dict:
    url_pattern = "http://localhost:{port}/v1/models/{model_name}:predict"

    url = url_pattern.format(port=conf.port, model_name=conf.model_name)

    response = requests.post(url, json=query, timeout=conf.timeout)
    response.raise_for_status()

    return response.json()


def error_result(query: Dict):
    result = {"data": [], "result": query["result"]}

    for item in query["data"]:
        result["data"].append(
            {
                "id": item["id"],
                "error": "Can't get results from model",
            }
        )

    return result


def result_postprocess(query: Dict, conf: OmegaConf) -> Dict:
    query["token"] = conf.token
    return query


def send_query(query: Dict, conf: OmegaConf) -> None:
    response = requests.post(conf.url_send, json=query, timeout=conf.timeout)
    response.raise_for_status()


def worker_step(conf: OmegaConf, logger: Logger) -> None:
    # Getting query from coordinator
    while True:
        try:
            query = get_task(conf.coordinator_access)
            break

        except Exception:
            logger.error(traceback.format_exc())
            logger.info("Error getting task from coordinator")
            time.sleep(conf.coordinator_access.coord_sleep_time)

    # If an empty query was returned, start a new loop
    if not query or query["result"] == 0:
        return

    for _ in range(conf.model_access.model_retries):
        try:
            result = model_calculation(query=query, conf=conf.model_access)
            break

        except Exception:
            logger.error(traceback.format_exc())
            logger.info("Error getting images")
            time.sleep(conf.model_access.model_sleep_time)

    # If model not responding, send error query
    else:
        logger.info(
            "The maximum number of attempts to get results from the model has been reached"
        )
        try:
            result = error_result(query)
        except Exception:
            logger.error(traceback.format_exc())
            logger.info("Error getting error query")
            return

    # Do some modifications before sending to coordinator
    try:
        output_query = result_postprocess(query=result, conf=conf.coordinator_access)

    except Exception:
        logger.error(traceback.format_exc())
        logger.info("Error postprocessing query")
        return

    # Sending result to coordinator
    for _ in range(conf.coordinator_access.sending_results_retries):
        try:
            send_query(query=output_query, conf=conf.coordinator_access)
            break
        except Exception:
            logger.error(traceback.format_exc())
            logger.info("Error sending results to coordinator")

    else:
        logger.info(
            "The maximum number of attempts to send results to the coordinator has been reached"
        )


def main() -> None:
    conf = OmegaConf.load("worker_config.yaml")
    logger = get_logger(__name__)

    while True:
        worker_step(conf, logger)


if __name__ == "__main__":
    main()
