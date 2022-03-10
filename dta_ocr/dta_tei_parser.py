import re
from dataclasses import dataclass
from typing import Dict, Optional
from bs4 import BeautifulSoup

PB_REGEX = re.compile(r"#f0*(\d+)")


@dataclass
class DTAPage:
    facsimile: int
    text: str


@dataclass
class DTADocument:
    pages: Dict[int, DTAPage]

    def get_page_text(self, facsimile: int) -> str:
        return self.pages[facsimile].text

    @staticmethod
    def from_tei_soup(soup: BeautifulSoup) -> Optional["DTADocument"]:
        return TEIParser(soup).parse()


class TEIParser:
    def __init__(self, soup: BeautifulSoup):
        self.soup = soup

    def parse(self) -> DTADocument:
        text = self.soup.find("text")
        pages: Dict[int, DTAPage] = {}
        facs = None

        for child in text.find_all(recursive=True):
            if child.name == "pb":
                match = PB_REGEX.match(child["facs"])
                facs = int(match.group(1))
                pages[facs] = DTAPage(facs, "")
            elif facs is not None and child.text:
                pages[facs].text += child.getText(separator=' ', strip=True)

        return DTADocument(pages)
