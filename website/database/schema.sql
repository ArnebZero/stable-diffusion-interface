DROP TABLE IF EXISTS requests;

CREATE TABLE requests (
    id TEXT PRIMARY KEY NOT NULL,
    stat INTEGER NOT NULL,
    edited TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
