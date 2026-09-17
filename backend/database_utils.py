import os
import sqlite3

DB_PATH = "results.db"


def create_table():
    if os.path.isfile(DB_PATH):
        return

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "CREATE TABLE tickets(id INTEGER PRIMARY_KEY, description TEXT, predicted_intent TEXT)"
    )

    con.commit()

    con.close()


def insert_into_table(description: str, intent: str):
    create_table()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "INSERT INTO tickets(description, predicted_intent) VALUES(?, ?)",
        (description, intent),
    )

    id = cur.lastrowid

    con.commit()

    con.close()

    return id
