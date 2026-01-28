#!/usr/bin/env bash

SEED=${DB_SEED:-0.42}

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE TABLE person_measurements (
        person_id         INTEGER PRIMARY KEY,
        age               INTEGER,
        bmi               FLOAT,
        systolic_bp       INTEGER,
        diastolic_bp      INTEGER,
        ldl_cholesterol   FLOAT,
        hba1c             FLOAT
    );

    SELECT setseed($SEED);

    INSERT INTO person_measurements
    SELECT
        gs AS person_id,
        20 + (random() * 60)::INT        AS age,
        18 + (random() * 15)             AS bmi,
        100 + (random() * 40)::INT       AS systolic_bp,
        60 + (random() * 25)::INT        AS diastolic_bp,
        70 + (random() * 120)            AS ldl_cholesterol,
        4.5 + (random() * 4)             AS hba1c
    FROM generate_series(1, 100) gs;
EOSQL
