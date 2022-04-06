PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS documents (
    dta_dirname TEXT PRIMARY KEY,
    page_count INTEGER CHECK ( page_count >= 0 ) NOT NULL,
    tei_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facsimiles (
    dta_dirname TEXT NOT NULL,
    page_number INTEGER CHECK ( page_number >= 1 ),
    status TEXT CHECK(
        status IN ( 'pending', 'error', 'finished' )
    ) NOT NULL,
    attempts INTEGER NOT NULL DEFAULT '0',
    error_msg TEXT CHECK (
        ( status = 'error' AND error_msg IS NOT NULL ) OR
        ( status != 'error' AND error_msg IS NULL )
    ),
    dta_url TEXT NOT NULL,
    hires_url TEXT,
    PRIMARY KEY ( dta_dirname, page_number ),
    FOREIGN KEY ( dta_dirname )
        REFERENCES documents ( dta_dirname )
            ON DELETE CASCADE
            ON UPDATE NO ACTION
);

CREATE TABLE IF NOT EXISTS segmentations (
    dta_dirname TEXT NOT NULL,
    page_number INTEGER CHECK ( page_number >= 1 ) NOT NULL,
    segmenter TEXT CHECK (
        ( status = 'finished' AND segmenter IS NOT NULL AND segmenter IN ( 'kraken', 'segmentation-pytorch' ) ) OR
        ( status != 'finished' AND segmenter IS NULL )
    ),
    model_path TEXT CHECK (
        ( status != 'finished' AND model_path IS NULL ) OR
        ( status = 'finished' AND segmenter = 'segmentation-pytorch' AND model_path IS NOT NULL ) OR
        ( status = 'finished' AND segmenter = 'kraken' AND model_path IS NULL )  
    ),
    file_path TEXT CHECK (
        ( status != 'finished' AND file_path IS NULL ) OR
        ( status = 'finished' AND file_path IS NOT NULL )
    ),
    status TEXT CHECK (
        status IN ( 'pending', 'error', 'finished' )
    ) NOT NULL,
    PRIMARY KEY ( dta_dirname, page_number ),
    FOREIGN KEY ( dta_dirname, page_number )
        REFERENCES facsimiles ( dta_dirname, page_number )
            ON DELETE CASCADE
            ON UPDATE NO ACTION
);

CREATE TABLE IF NOT EXISTS predictions (
    dta_dirname TEXT NOT NULL,
    page_number INTEGER CHECK ( page_number >= 1 ) NOT NULL,
    prediction_path TEXT CHECK (
        ( status != 'finished' AND prediction_path IS NULL ) OR
        ( status = 'finished' AND prediction_path IS NOT NULL )
    ),
    status TEXT CHECK (
        status IN ( 'pending', 'error', 'finished' )
    ) NOT NULL,
    PRIMARY KEY ( dta_dirname, page_number ),
    FOREIGN KEY ( dta_dirname, page_number )
        REFERENCES segmentations ( dta_dirname, page_number )
            ON DELETE CASCADE
            ON UPDATE NO ACTION
);

CREATE TABLE IF NOT EXISTS matchings (
    dta_dirname TEXT NOT NULL,
    page_number INTEGER CHECK ( page_number >= 1 ) NOT NULL,
    gt_path TEXT CHECK (
        ( status != 'finished' AND gt_path IS NULL ) OR
        ( status = 'finished' AND gt_path IS NOT NULL )
    ),
    match_ratio REAL CHECK (
        ( status = 'finished' AND match_ratio IS NOT NULL ) OR
        ( status != 'finished' AND match_ratio IS NULL )
    ),
    status TEXT CHECK (
        status IN ( 'pending', 'error', 'finished' )
    ) NOT NULL,
    PRIMARY KEY ( dta_dirname, page_number ),
    FOREIGN KEY ( dta_dirname, page_number )
        REFERENCES predictions ( dta_dirname, page_number )
            ON DELETE CASCADE
            ON UPDATE NO ACTION
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_pkey_index ON documents ( dta_dirname );
CREATE UNIQUE INDEX IF NOT EXISTS facsimiles_pkey_index ON facsimiles ( dta_dirname, page_number );
CREATE UNIQUE INDEX IF NOT EXISTS segmentations_pkey_index ON segmentations ( dta_dirname, page_number );
CREATE UNIQUE INDEX IF NOT EXISTS predictions_pkey_index ON predictions ( dta_dirname, page_number );
CREATE UNIQUE INDEX IF NOT EXISTS matchings_pkey_index ON matchings ( dta_dirname, page_number );