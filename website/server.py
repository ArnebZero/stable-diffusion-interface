import os
import sqlite3
import zipfile
from datetime import datetime as dt
from secrets import token_hex

import numpy as np
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from PIL import Image

app = Flask(__name__)
app.secret_key = b"1234567890qwertyuiopasdfghjklzxcvbnm"

TOKEN = "mytoken"


def get_db_connection():
    conn = sqlite3.connect("database/database.db")
    conn.row_factory = sqlite3.Row
    return conn


def create_zip():
    if "username" not in session:
        session["username"] = token_hex(8)

    user_id = session["username"]

    if os.path.exists(f"user_data/{user_id}/images.zip"):
        return

    file_names = []

    for ind in range(3):
        if not os.path.exists(f"user_data/{user_id}/img_{ind}.png"):
            raise RuntimeError("No files")

        file_names.append(f"user_data/{user_id}/img_{ind}.png")

    if not os.path.exists(f"user_data/{user_id}/text.txt"):
        raise RuntimeError("No files")

    file_names.append(f"user_data/{user_id}/text.txt")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM requests WHERE id = ? AND stat=3 LIMIT 1",
        (session["username"],),
    )
    data = cur.fetchone()
    conn.close()

    if not data:
        raise RuntimeError("Invalid status")

    zip_file = zipfile.ZipFile(f"user_data/{user_id}/images.zip", mode="w")

    try:
        for file_name in file_names:
            zip_file.write(file_name, os.path.basename(file_name))

        zip_file.close()

    except:
        zip_file.close()
        if os.path.exists(f"user_data/{user_id}/images.zip"):
            os.remove(f"user_data/{user_id}/images.zip")


@app.route("/", methods=["GET"])
def start_page():
    return render_template("start_page.html")


@app.errorhandler(500)
def server_error(error):
    return render_template("500_error.html"), 500


@app.errorhandler(404)
def server_error(error):
    return render_template("404_error.html"), 404


@app.route("/get_results", methods=["GET"])
def get_results():
    if "username" not in session:
        session["username"] = token_hex(8)

    text = request.args.get("text", type=str)

    # submit task if possible
    if text:
        assert len(text) <= 100
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT stat FROM requests WHERE id = ? LIMIT 1",
            (session["username"],),
        )
        data = cur.fetchone()
        if data is None or (data["stat"] != 1 and data["stat"] != 2):
            os.makedirs(f"user_data/{session['username']}", exist_ok=True)
            with open(
                f"user_data/{session['username']}/text.txt", "w", encoding="utf-8"
            ) as fp:
                fp.write(text)

            if data is None:
                cur.execute(
                    "INSERT INTO requests (id, stat, edited) VALUES (?, ?, ?)",
                    (session["username"], 1, dt.now()),
                )
            else:
                cur.execute(
                    "UPDATE requests SET stat=1, edited=? WHERE id=?",
                    (dt.now(), session["username"]),
                )
            conn.commit()
            conn.close()

            return redirect("/get_results")
        else:
            conn.close()
            return render_template("too_many_queries.html")

    # return images if exists
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM requests WHERE id = ? LIMIT 1",
        (session["username"],),
    )

    cur.execute(
        "SELECT * FROM requests WHERE id = ? AND stat=3 LIMIT 1",
        (session["username"],),
    )
    data = cur.fetchone()

    flag = True
    if data is not None:
        for ind in range(3):
            if not os.path.isfile(f"user_data/{session['username']}/img_{ind}.png"):
                flag = False
                break
        if not os.path.isfile(f"user_data/{session['username']}/text.txt"):
            flag = False
    else:
        flag = False

    if flag:
        with open(f"user_data/{session['username']}/text.txt", "r") as fp:
            text = fp.read()

        return render_template(
            "results.html",
            QUERY=text,
            IMAGE_1=url_for("download_image", filepath="img_0.png"),
            IMAGE_2=url_for("download_image", filepath="img_1.png"),
            IMAGE_3=url_for("download_image", filepath="img_2.png"),
        )

    conn.close()

    return render_template("loading_results.html")


@app.route("/images/<path:filepath>", methods=["GET"])
def download_image(filepath):
    if "username" not in session:
        session["username"] = token_hex(8)
    return send_from_directory(
        f"user_data/{session['username']}", filepath, as_attachment=True
    )


@app.route("/get_images", methods=["GET"])
def download_images_zip():
    if "username" not in session:
        session["username"] = token_hex(8)

    create_zip()

    return send_from_directory(
        f"user_data/{session['username']}", "images.zip", as_attachment=True
    )


@app.route("/api/v1/get_task", methods=["POST"])
def get_task():
    content = request.json

    if content is None:
        return {"error": "No data"}, 400

    if "token" not in content:
        return {"error": "Not authorized"}, 401

    if content["token"] != TOKEN:
        return {"error": "No valid authentication credentials"}, 401

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM requests WHERE stat = 1 LIMIT 3")
    data = cur.fetchall()

    ids = [item["id"] for item in data]

    output_data = []
    for user_id in ids:
        if os.path.isfile(f"user_data/{user_id}/text.txt"):
            with open(f"user_data/{user_id}/text.txt", "r", encoding="utf-8") as fp:
                text = fp.read()
            output_data.append({"id": user_id, "text": text})

    if not output_data:
        conn.close()
        return {"result": 0}

    for item in output_data:
        cur.execute(
            "UPDATE requests SET stat=2, edited=? WHERE id=?", (dt.now(), item["id"])
        )

    conn.commit()
    conn.close()

    return {"result": len(output_data), "data": output_data}


@app.route("/api/v1/send_task", methods=["POST"])
def send_task():
    content = request.json

    if content is None:
        return {"error": "No data"}, 400

    if "token" not in content:
        return {"error": "Not authorized"}, 401

    if content["token"] != TOKEN:
        return {"error": "No valid authentication credentials"}, 401

    if content["result"] == 0:
        return {}, 201

    conn = get_db_connection()
    cur = conn.cursor()

    error_users = []
    update_users = []
    for item in content["data"]:
        user_id = item["id"]
        if not os.path.isdir(f"user_data/{user_id}"):
            continue

        cur.execute("SELECT * FROM requests WHERE id=? AND stat=2 LIMIT 1", (user_id,))
        data = cur.fetchone()

        if not data:
            continue

        if "error" in item:
            error_users.append(user_id)
            continue

        images = item["images"]

        images = [
            np.array(images[str(ind)], dtype=np.uint8) for ind in range(len(images))
        ]
        images = [img.reshape((512, 512, 3)) for img in images]
        images = [Image.fromarray(img) for img in images]

        for ind, img in enumerate(images):
            img.save(f"user_data/{user_id}/img_{ind}.png")

        if os.path.exists(f"user_data/{user_id}/images.zip"):
            os.remove(f"user_data/{user_id}/images.zip")

        update_users.append(user_id)

    for user_id in update_users:
        cur.execute(
            "UPDATE requests SET stat=3, edited=? WHERE id=?", (dt.now(), user_id)
        )

    for user_id in error_users:
        cur.execute(
            "UPDATE requests SET stat=4, edited=? WHERE id=?", (dt.now(), user_id)
        )

    conn.commit()
    conn.close()

    return {}, 201


@app.route("/api/v1/ready", methods=["GET"])
def ready_user():
    if "username" not in session:
        session["username"] = token_hex(8)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT stat FROM requests WHERE id = ? LIMIT 1",
        (session["username"],),
    )
    data = cur.fetchone()

    conn.close()

    if data is not None:
        return {"status": data["stat"]}

    else:
        return {"status": 0}


@app.route("/error_500", methods=["GET"])
def check_500():
    return render_template("500_error.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
