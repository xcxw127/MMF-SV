import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Region:
    """
    A region from a bed file.

    Coordinates are 0-based, half open
    """
    contig: str
    start: int
    end: int

    @classmethod
    def from_bed_line(cls, line: str) -> Optional['Region']:
        try:
            contig, start, end = line.split('\t')[:3]
            return Region(
                contig=contig,
                start=int(start),
                end=int(end),
            )
        except ValueError as ex:
            log.warning(f'Invalid region line: {line} ({ex})')
            return None

    def __str__(self) -> str:
        return f'{self.contig}:{self.start}-{self.end}'
