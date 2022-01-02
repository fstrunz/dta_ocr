import asyncio
import argparse
import aiosqlite


async def create_download_list_tables(download_list_db: str):
    async with aiosqlite.connect(download_list_db) as db:
        await db.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                dta_dirname TEXT PRIMARY KEY,
                page_count INTEGER CHECK ( page_count >= 0 ) NOT NULL
            );"""
        )
        await db.execute(
            """CREATE TABLE IF NOT EXISTS facsimiles (
                dta_dirname INTEGER,
                page_number INTEGER CHECK ( page_number >= 1 ),
                status TEXT CHECK(
                    status IN ( 'pending', 'error', 'finished' )
                ) NOT NULL,
                attempts INTEGER NOT NULL DEFAULT '0',
                error_msg TEXT CHECK (
                    ( status = 'error' AND error_msg IS NOT NULL ) OR
                    ( status != 'error' AND error_msg IS NULL )
                ),
                PRIMARY KEY (dta_dirname, page_number),
                FOREIGN KEY (dta_dirname)
                    REFERENCES documents (dta_dirname)
                        ON DELETE CASCADE
                        ON UPDATE NO ACTION
            );"""
        )
        await db.commit()


async def main():
    arg_parser = argparse.ArgumentParser(
        description=(
            "Given an extracted copy of the complete DTA corpus, " +
            "download the corresponding facsimiles. Such a corpus " +
            "can be downloaded at https://www.deutschestextarchiv.de/download."
        )
    )
    arg_parser.add_argument(
        "--corpus-dir", dest="corpus_dir", required=True,
        help="The extracted DTA corpus directory to use as input."
    )
    arg_parser.add_argument(
        "--output-dir", dest="output_dir", default="facsimiles",
        help=(
            "The directory to write the downloaded facsimiles to. The " +
            "images will be stored within this directory in folders named " +
            "after the document's DTA dirname, in which the images have " +
            "their page numbers as names."
        )
    )
    arg_parser.add_argument(
        "--download-list", dest="download_list", default="downloaded.db",
        help=(
            "The location of a SQLite database which stores which " +
            "facsimiles have already been successfully downloaded. This is " +
            "used for resuming the download after an intentional or " +
            "unintentional disruption.\n" +
            "The database will be created if it does not exist!"
        )
    )
    args = arg_parser.parse_args()
    await create_download_list_tables(args.download_list)

if __name__ == "__main__":
    asyncio.run(main())
