"""
Parsel lets you extract text from XML/HTML documents using XPath
or CSS selectors
"""

__author__ = "Scrapy project"
__email__ = "info@scrapy.org"
__version__ = "1.12.1"
__all__ = [
    "AbsoluteXPathWarning",
    "ExtractionResult",
    "Field",
    "FieldDefinitionError",
    "FieldDiagnostics",
    "Record",
    "RecordExtractionError",
    "Selector",
    "SelectorList",
    "css2xpath",
    "extract_records",
    "xpathfuncs",
]

from parsel import xpathfuncs
from parsel.csstranslator import css2xpath
from parsel.records import (
    ExtractionResult,
    Field,
    FieldDefinitionError,
    FieldDiagnostics,
    Record,
    RecordExtractionError,
    extract_records,
)
from parsel.selector import AbsoluteXPathWarning, Selector, SelectorList

xpathfuncs.setup()
