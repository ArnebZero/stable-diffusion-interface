import sqlite3

connection = sqlite3.connect('database.db')

cur = connection.cursor()
cur.execute("SELECT * FROM requests")

rows = cur.fetchall()

for row in rows:
    print(row)
