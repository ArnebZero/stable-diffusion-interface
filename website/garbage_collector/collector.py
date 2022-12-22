import os
import time
import shutil
import sqlite3
import traceback
from datetime import datetime, timedelta

from logger import get_logger

TIMEOUT_MINUTES = 30


def get_db_connection():
    conn = sqlite3.connect("../database/database.db")
    conn.row_factory = sqlite3.Row
    return conn

def mark_rows():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE requests SET stat=5, edited=? WHERE edited < ?", 
        (
            datetime.now(),
            datetime.now() - timedelta(minutes=TIMEOUT_MINUTES)
        )
    )

    conn.commit()
    conn.close()

def remove_marked_rows(logger):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM requests WHERE stat=5")

    data = cur.fetchall()
    rows_for_delete = [item["id"] for item in data]

    if rows_for_delete:
        cur.execute("DELETE FROM requests WHERE stat=5")

    conn.commit()

    for folder in rows_for_delete:
        shutil.rmtree(f"../user_data/{folder}")
        logger.info("Remove marked session %s", folder)

    conn.close()

def remove_not_existed_rows(logger):
    dirs = os.listdir("../user_data/")

    conn = get_db_connection()
    cur = conn.cursor()

    for dir_name in dirs:
        cur.execute("SELECT * FROM requests WHERE id=? LIMIT 1", (dir_name, ))
        data = cur.fetchone()

        if not data:
            shutil.rmtree(f"../user_data/{dir_name}")
            logger.info("Remove not existed session %s", dir_name)

    conn.close()

logger = get_logger(__name__)

while True:
    try:
        remove_not_existed_rows(logger)
    except:
        logger.error(traceback.format_exc())
        logger.info("Error removing nonexisting rows")

    try:
        remove_marked_rows(logger)
    except:
        logger.error(traceback.format_exc())
        logger.info("Error removing marked rows")

    try:
        mark_rows()
    except:
        logger.error(traceback.format_exc())
        logger.info("Error marking rows")
    
    time.sleep(TIMEOUT_MINUTES * 60)
