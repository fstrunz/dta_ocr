from typing import Dict, List, Optional, Tuple
import asyncio
import argparse
import aiosqlite
import aiohttp
import mimetypes
import traceback
import re
import collections
from aiofile import async_open
from lxml import etree
from pathlib import Path
from dataclasses import dataclass

TEI_NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}
FACSIMILES_HTML_URL = "https://www.deutschestextarchiv.de/book/images/{}"
ANCHOR_PATTERN = re.compile(
    r"https://www\.deutschestextarchiv\.de/book/view/(.+)\?p=(\d+)"
)


async def dirname_from_tei(tei_file: Path) -> Optional[str]:
    # async with async_open(tei_file, "rb") as file:
    #     try:
    #         root = etree.fromstring(await file.read())
    #     except etree.XMLSyntaxError:
    #         print(f"TEI file {tei_file} has malformed XML!")
    #         return None

    # idnos = root.xpath(
    #     ".//tei:idno[@type='DTADirName']",
    #     namespaces=TEI_NAMESPACES
    # )

    # if not idnos:
    #    print(
    #        f"No idno tag with type='DTADirName' found in TEI file " +
    #        f"{tei_file}."
    #    )
    #    return None

    # return idnos[0].text

    filename = tei_file.name
    dot_index = filename.index(".")
    return filename[:dot_index]


@dataclass(frozen=True, eq=True)
class Facsimile:
    dta_dirname: str
    page_num: int
    dta_url: str
    hires_url: Optional[str]
    attempts: int


async def find_available_facsimiles(
    http: aiohttp.ClientSession, dta_dirname: str
) -> List[Facsimile]:
    print(f"Finding available facsimiles for {dta_dirname}...")
    async with http.get(FACSIMILES_HTML_URL.format(dta_dirname)) as response:
        html = await response.text()
        root = etree.fromstring(html, parser=etree.HTMLParser())
        anchors = root.xpath(
            ".//td[@id='content-middle']//div/a[@href[starts-with(.," +
            "'https://www.deutschestextarchiv.de/book/view/')]]"
        )

        facsimiles = []

        for anchor in anchors:
            href = anchor.get("href")
            m = ANCHOR_PATTERN.match(href)

            if m is None:
                print(
                    f"Href {href} did not look as expected, ignoring anchor..."
                )
                continue

            try:
                dirname, page_num = m.group(1), int(m.group(2))
            except ValueError:
                print("Anchor href did not contain a valid page number!")
                continue

            if dta_dirname != dirname:
                print("Anchor href dirname did not match the given dirname!")
                continue

            facsimiles.append(Facsimile(dta_dirname, page_num, href, None, 0))

        return facsimiles


async def write_facsimile_to_download_list(
    http: aiohttp.ClientSession, db: aiosqlite.Connection, dta_dirname: str
):
    cursor = await db.execute(
        "SELECT page_count FROM documents WHERE dta_dirname = ?",
        [dta_dirname]
    )

    row = await cursor.fetchone()
    if row is None:
        facsimiles = await find_available_facsimiles(
            http, dta_dirname
        )

        await db.execute(
            """INSERT OR IGNORE INTO documents ( dta_dirname, page_count )
            VALUES ( ?, ? );""", [dta_dirname, len(facsimiles)]
        )

        for facsimile in facsimiles:
            await db.execute(
                """INSERT OR IGNORE INTO facsimiles
                    ( dta_dirname, page_number,
                    status, attempts, error_msg,
                    dta_url, hires_url )
                VALUES
                    ( ?, ?, ?, ?, NULL, ?, ? );""",
                [dta_dirname, facsimile.page_num, "pending",
                    facsimile.attempts, facsimile.dta_url,
                    facsimile.hires_url]
            )

        await db.commit()


async def find_pending_facsimiles(
    http: aiohttp.ClientSession, download_list_db: Path, corpus_path: Path
) -> List[Facsimile]:
    async with aiosqlite.connect(download_list_db) as db:
        print("Collecting DTA dirnames from TEI files...")
        dta_dirnames = [
            await dirname_from_tei(path) for path in corpus_path.iterdir()
            if path.is_file() and path.name.endswith(".TEI-P5.xml")
        ]

        for f in asyncio.as_completed([
            write_facsimile_to_download_list(http, db, dta_dirname)
            for dta_dirname in dta_dirnames
            if dta_dirname is not None
        ]):
            try:
                await f
            except Exception:
                traceback.print_exc()

        print("Computing the remaining facsimiles...")
        async with db.execute_fetchall(
            """SELECT dta_dirname, page_number, dta_url, hires_url, attempts
            FROM facsimiles
            WHERE status != 'finished';"""
        ) as cursor:
            return [
                Facsimile(
                    dta_dirname, page_number, dta_url, hires_url, attempts
                )
                for dta_dirname, page_number, dta_url,
                hires_url, attempts in cursor
            ]


async def fetch_all_hires_urls_from_db(
    db: aiosqlite.Connection
) -> Dict[str, Dict[int, str]]:
    hires_dict = {}

    async with db.execute(
        """SELECT dta_dirname, page_number, hires_url FROM facsimiles
        WHERE hires_url IS NOT NULL"""
    ) as cursor:
        async for dta_dirname, page_number, hires_url in cursor:
            if dta_dirname not in hires_dict:
                hires_dict[dta_dirname] = {}
            hires_dict[dta_dirname][page_number] = hires_url

    return hires_dict


async def find_hires_url(
    http: aiohttp.ClientSession, db: aiosqlite.Connection,
    facsimile: Facsimile, hires_dict: Dict[str, Dict[int, str]]
) -> Optional[str]:
    hires_url = None

    if facsimile.dta_dirname in hires_dict:
        if facsimile.page_num in hires_dict[facsimile.dta_dirname]:
            hires_url = hires_dict[facsimile.dta_dirname][facsimile.page_num]

    if hires_url is None:
        href = facsimile.dta_url
        async with http.get(href) as response:
            html = await response.text()
            root = etree.fromstring(html, parser=etree.HTMLParser())

            hires_anchors = root.xpath(
                ".//div[@class='rightcol']/ul/li/a[@href[starts-with(.," +
                "'https://media.dwds.de/dta/images')]]"
            )

            if not hires_anchors:
                print("No anchor for high resolution facsimile found!")
                return None

            # there should only be one of these
            hires_anchor = hires_anchors[0]
            hires_url = hires_anchor.get("href")

        await db.execute(
            """UPDATE facsimiles SET hires_url = ?
            WHERE dta_dirname = ? AND page_number = ?;""",
            [hires_url, facsimile.dta_dirname, facsimile.page_num]
        )
        await db.commit()

    return hires_url


async def write_error_to_download_list(
    db: aiosqlite.Connection, facsimile: Facsimile, error_msg: str
):
    await db.execute(
        """UPDATE facsimiles
            SET
                status = 'error',
                error_msg = ?,
                attempts = attempts + 1
            WHERE dta_dirname = ? AND page_number = ?;""",
        [error_msg, facsimile.dta_dirname, facsimile.page_num]
    )
    await db.commit()


async def write_success_to_download_list(
    db: aiosqlite.Connection, facsimile: Facsimile
):
    await db.execute(
        """UPDATE facsimiles
            SET
                status = 'finished',
                error_msg = NULL,
                attempts = attempts + 1
            WHERE dta_dirname = ? AND page_number = ?;""",
        [facsimile.dta_dirname, facsimile.page_num]
    )
    await db.commit()


async def download_facsimile(
    http: aiohttp.ClientSession, db: aiosqlite.Connection,
    facsimile: Facsimile, hires_dict: Dict[str, Dict[int, str]]
) -> Tuple[Facsimile, bytes, str]:
    try:
        hires_url = await find_hires_url(http, db, facsimile, hires_dict)
    except Exception:
        print(
            "Failed to find hires URL for " +
            f"{facsimile.dta_dirname}/{facsimile.page_num}!!!"
        )
        traceback.print_exc()
        await write_error_to_download_list(
            db, facsimile, "could not find hires url"
        )
        return None

    try:
        async with http.get(hires_url) as response:
            content_type = response.headers["content-type"]

            extension = mimetypes.guess_extension(content_type, strict=True)
            if extension is None or extension == ".jpe":
                extension = ".jpg"

            await write_success_to_download_list(db, facsimile)
            return facsimile, await response.read(), extension
    except Exception:
        print(
            "Failed to download facsimile " +
            f"{facsimile.dta_dirname}/{facsimile.page_num}!!!"
        )
        traceback.print_exc()
        await write_error_to_download_list(
            db, facsimile, "download failed"
        )
        return None


async def create_download_list_tables(db: aiosqlite.Connection):
    await db.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            dta_dirname TEXT PRIMARY KEY,
            page_count INTEGER CHECK ( page_count >= 0 ) NOT NULL
        );"""
    )
    await db.execute(
        """CREATE TABLE IF NOT EXISTS facsimiles (
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
                    ON UPDATE NO ACTION,
            UNIQUE ( dta_dirname, page_number )
        );"""
    )
    await db.commit()


async def main():
    arg_parser = argparse.ArgumentParser(
        "download_facsimiles",
        description=(
            "Given an extracted copy of the complete DTA corpus, " +
            "download the corresponding facsimiles. Such a corpus " +
            "can be downloaded at https://www.deutschestextarchiv.de/download."
        )
    )
    arg_parser.add_argument(
        "--coroutine-count", dest="coroutine_count", default=4, type=int,
        help="The amount of HTTP requests to make asynchronously."
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

    corpus_path = Path(args.corpus_dir)
    if not corpus_path.is_dir():
        print("Given corpus directory is not a directory!")
        exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    download_list_db = Path(args.download_list)

    async with aiosqlite.connect(download_list_db) as db:
        await create_download_list_tables(db)

        async with db.execute(
            """SELECT COUNT(dta_dirname) FROM facsimiles
               NATURAL JOIN documents
               WHERE facsimiles.status = 'finished'
               GROUP BY dta_dirname
               HAVING COUNT(dta_dirname) = documents.page_count"""
        ) as cursor:
            count = 0
            fac_count = 0
            async for c in cursor:
                count += 1
                fac_count += c[0]

            print(
                f"So far, {count} documents " +
                f"have been fully downloaded, encompassing {fac_count} " +
                "facsimiles."
            )

        hires_dict = await fetch_all_hires_urls_from_db(db)

        async with aiohttp.ClientSession() as http:
            pending = await find_pending_facsimiles(
                http, download_list_db, corpus_path
            )

        tasks = collections.deque(pending)

        async def worker():
            async with aiohttp.ClientSession() as http:
                while tasks:
                    facsimile = tasks.pop()

                    result = await download_facsimile(
                        http, db, facsimile, hires_dict
                    )

                    if result is None:
                        continue

                    fac, data, ext = result

                    doc_dir = output_path / Path(fac.dta_dirname)
                    doc_dir.mkdir(exist_ok=True)

                    file_path = doc_dir / str(f"{fac.page_num}{ext}")
                    async with async_open(file_path, "wb") as file:
                        print(f"Writing {file_path}...")
                        await file.write(data)

        await asyncio.gather(
            *[worker() for _ in range(args.coroutine_count)]
        )

if __name__ == "__main__":
    asyncio.run(main())
