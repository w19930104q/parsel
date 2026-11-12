"""One-call record extraction from HTML and XML documents.

This module builds on :class:`~parsel.Selector` to cover the common case of
extracting a list of repeated records (products, entries, search results, ...)
together with their fields in a single call.

The :func:`extract_records` function first locates the repeated records using
a record selector, then evaluates one :class:`Field` per business field
against each record. Every field supports text and attribute extraction,
multi-value hits, relative-URL normalization and the XML default namespace
without the caller having to iterate over selectors manually. Missing
required fields never pass silently: they raise
:class:`RecordExtractionError`, which still carries the values and the
diagnostics collected for every record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias
from urllib.parse import urljoin as _urljoin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from parsel.selector import Selector, SelectorList

#: Prefix under which the default namespace of an XML document is registered
#: when the document root declares one (``xmlns="..."``).
DEFAULT_NAMESPACE_PREFIX = "d"

#: Values extracted for a field: a single string, several strings, or
#: ``None`` when an optional field has no hit.
FieldValue: TypeAlias = "str | list[str] | None"


class FieldDefinitionError(ValueError):
    """Raised when a :class:`Field` or an :func:`extract_records` call is
    configured incorrectly (for example, neither or both of ``xpath`` and
    ``css`` are given)."""


@dataclass(frozen=True, slots=True)
class Field:
    """Description of one business field to extract from each record.

    Exactly one of ``xpath`` and ``css`` selects the field nodes relatively
    to a record. By default the serialized nodes (text, attribute value or
    element markup, depending on the expression) are returned as strings.

    ``attr`` additionally reads the given attribute from the matched nodes,
    so ``Field(css="a", attr="href")`` and ``Field(xpath=".//a/@href")``
    are equivalent.

    ``required`` marks a field that must hit on every record. A record on
    which a required field is missing makes :func:`extract_records` raise
    :class:`RecordExtractionError`; optional fields yield ``None`` (or an
    empty list when ``multi`` is set).

    ``multi`` makes the field collect every hit as a list instead of only
    the first one.

    ``url`` normalizes each hit as a URL: relative references are joined
    with the document base URL (see :func:`extract_records`), while
    absolute URLs are returned unchanged.
    """

    xpath: str | None = None
    css: str | None = None
    attr: str | None = None
    required: bool = False
    multi: bool = False
    url: bool = False

    def __post_init__(self) -> None:
        if (self.xpath is None) == (self.css is None):
            raise FieldDefinitionError(
                "Field requires exactly one of the xpath and css arguments, "
                f"got xpath={self.xpath!r}, css={self.css!r}"
            )
        if self.attr is not None and not self.attr:
            raise FieldDefinitionError("attr must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class Record:
    """The fields extracted from one repeated record.

    ``index`` is the zero-based position of the record among the records
    selected from the document.
    """

    index: int
    values: dict[str, FieldValue]
    missing_required: tuple[str, ...] = ()

    def get(self, name: str, default: Any = None) -> Any:
        """Return the value of field *name*, or *default* when absent."""
        return self.values.get(name, default)


@dataclass(frozen=True, slots=True)
class FieldDiagnostics:
    """Per-field extraction statistics across all records.

    ``total_hits`` is the number of matched nodes over every record.
    ``hit_records`` and ``missing_records`` are the indexes of records on
    which the field respectively matched at least once or did not match.
    """

    total_hits: int = 0
    hit_records: tuple[int, ...] = ()
    missing_records: tuple[int, ...] = ()

    @property
    def hit_count(self) -> int:
        """Number of records on which the field matched at least once."""
        return len(self.hit_records)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Everything :func:`extract_records` found in a document."""

    record_count: int
    field_diagnostics: Mapping[str, FieldDiagnostics]
    records: tuple[Record, ...]

    @property
    def values(self) -> tuple[dict[str, FieldValue], ...]:
        """The field values of every record, in document order."""
        return tuple(record.values for record in self.records)

    @property
    def missing_required(self) -> tuple[tuple[int, str], ...]:
        """``(record index, field name)`` pairs for missing required fields."""
        return tuple(
            (record.index, name)
            for record in self.records
            for name in record.missing_required
        )


class RecordExtractionError(Exception):
    """Raised when at least one required field is missing from a record.

    The partially extracted data is not lost: ``result`` holds the full
    :class:`ExtractionResult`, including the values that were found, and
    ``missing`` lists the ``(record index, field name)`` pairs.
    """

    def __init__(
        self,
        result: ExtractionResult,
        missing: Sequence[tuple[int, str]],
    ) -> None:
        self.result = result
        self.missing = tuple(missing)
        details = ", ".join(
            f"record {index}: {', '.join(names)}"
            for index, names in _group_missing(self.missing)
        )
        super().__init__(f"Missing required field(s): {details}")


def _group_missing(
    missing: Sequence[tuple[int, str]],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    grouped: dict[int, list[str]] = {}
    for index, name in missing:
        grouped.setdefault(index, []).append(name)
    return tuple((index, tuple(names)) for index, names in sorted(grouped.items()))


def _default_namespace(selector: Selector) -> str | None:
    nsmap = getattr(selector.root, "nsmap", None)
    if not isinstance(nsmap, dict):
        return None
    uri = nsmap.get(None)
    return uri if isinstance(uri, str) else None


def _build_namespaces(
    selector: Selector, namespaces: Mapping[str, str] | None
) -> dict[str, str]:
    merged: dict[str, str] = {}
    default_uri = _default_namespace(selector)
    if default_uri is not None:
        merged[DEFAULT_NAMESPACE_PREFIX] = default_uri
    if namespaces:
        merged.update(namespaces)
    return merged


def _resolve_base_url(selector: Selector, base_url: str | None) -> str | None:
    if base_url is not None:
        return base_url
    root = selector.root
    return getattr(root, "base", None)


def _select_nodes(
    selector: Selector, field: Field, namespaces: Mapping[str, str]
) -> SelectorList[Selector]:
    if field.css is not None:
        query = selector._css2xpath(field.css)
        return selector.xpath(query, namespaces=namespaces)
    xpath_query = field.xpath
    if xpath_query is None:  # pragma: no cover - ruled out by Field validation
        raise FieldDefinitionError("Field has neither xpath nor css")
    return selector.xpath(xpath_query, namespaces=namespaces)


def _extract_field(
    selector: Selector,
    field: Field,
    namespaces: Mapping[str, str],
    base_url: str | None,
) -> tuple[FieldValue, int]:
    nodes = _select_nodes(selector, field, namespaces)
    values: list[str]
    if field.attr is not None:
        values = []
        for node in nodes:
            value = node.attrib.get(field.attr)
            if value is not None:
                values.append(value)
    else:
        values = nodes.getall()
    hit_count = len(values)
    if field.url and values:
        if base_url is None:
            raise ValueError(
                "url=True requires a base URL: pass base_url to extract_records() "
                "or parse the document with Selector(base_url=...)"
            )
        values = [_urljoin(base_url, value) for value in values]
    if not values:
        return ([] if field.multi else None), hit_count
    if field.multi:
        return values, hit_count
    return values[0], hit_count


def _extract_record(
    index: int,
    record: Selector,
    fields: Mapping[str, Field],
    namespaces: Mapping[str, str],
    base_url: str | None,
) -> tuple[Record, Mapping[str, int]]:
    values: dict[str, FieldValue] = {}
    missing_required: list[str] = []
    hit_counts: dict[str, int] = {}
    for name, field in fields.items():
        value, hit_count = _extract_field(record, field, namespaces, base_url)
        values[name] = value
        hit_counts[name] = hit_count
        if field.required and not hit_count:
            missing_required.append(name)
    return (
        Record(index=index, values=values, missing_required=tuple(missing_required)),
        hit_counts,
    )


def _build_diagnostics(
    names: Sequence[str],
    hit_counts: Sequence[Mapping[str, int]],
) -> dict[str, FieldDiagnostics]:
    diagnostics: dict[str, FieldDiagnostics] = {}
    for name in names:
        total_hits = sum(counts[name] for counts in hit_counts)
        hit_records = tuple(
            index for index, counts in enumerate(hit_counts) if counts[name]
        )
        missing_records = tuple(
            index for index, counts in enumerate(hit_counts) if not counts[name]
        )
        diagnostics[name] = FieldDiagnostics(
            total_hits=total_hits,
            hit_records=hit_records,
            missing_records=missing_records,
        )
    return diagnostics


def _select_records(
    selector: Selector,
    record_xpath: str | None,
    record_css: str | None,
    namespaces: Mapping[str, str],
) -> SelectorList[Selector]:
    if record_css is not None:
        return selector.xpath(selector._css2xpath(record_css), namespaces=namespaces)
    if record_xpath is None:  # pragma: no cover - ruled out by validation
        raise FieldDefinitionError("record_xpath is required")
    return selector.xpath(record_xpath, namespaces=namespaces)


def _extract_all_records(
    records: SelectorList[Selector],
    fields: Mapping[str, Field],
    namespaces: Mapping[str, str],
    base_url: str | None,
) -> tuple[tuple[Record, ...], tuple[Mapping[str, int], ...]]:
    extracted: list[Record] = []
    hit_counts: list[Mapping[str, int]] = []
    for index, record in enumerate(records):
        record_data, counts = _extract_record(
            index, record, fields, namespaces, base_url
        )
        extracted.append(record_data)
        hit_counts.append(counts)
    return tuple(extracted), tuple(hit_counts)


def extract_records(
    selector: Selector,
    fields: Mapping[str, Field],
    *,
    record_xpath: str | None = None,
    record_css: str | None = None,
    base_url: str | None = None,
    namespaces: Mapping[str, str] | None = None,
) -> ExtractionResult:
    """Extract repeated records and their fields from *selector* in one call.

    ``fields`` maps business field names to :class:`Field` instances. The
    repeated records are located with exactly one of ``record_xpath`` and
    ``record_css``, and each field expression is evaluated relatively to a
    record.

    ``base_url`` is used to normalize fields with ``url=True``. It
    defaults to the base URL of the parsed document, if any (the
    ``base_url`` argument of :class:`~parsel.Selector` or an HTML
    ``<base>`` tag).

    ``namespaces`` is an optional ``prefix: namespace-uri`` mapping added on
    top of those registered on the selector. When an XML document declares a
    default namespace (``xmlns="..."``), it is automatically registered
    under the :data:`DEFAULT_NAMESPACE_PREFIX` (``"d"``) prefix, so XPath
    expressions can write ``d:entry`` instead of registering the namespace
    manually.

    Invalid XPath or CSS expressions propagate the same exceptions as
    :meth:`Selector.xpath` and :meth:`Selector.css`. Missing required fields
    raise :class:`RecordExtractionError`; the returned (and attached)
    :class:`ExtractionResult` always contains per-record values and
    per-field hit counts.

    >>> from parsel import Selector
    >>> from parsel.records import Field, extract_records
    >>> sel = Selector(text='''
    ... <ul>
    ...   <li><a href="/p/1">First</a><span class="tag">a</span><span class="tag">b</span></li>
    ...   <li><a href="/p/2">Second</a></li>
    ... </ul>''', base_url="https://example.org/")
    >>> result = extract_records(
    ...     sel,
    ...     record_css="li",
    ...     fields={
    ...         "name": Field(css="a::text", required=True),
    ...         "link": Field(css="a", attr="href", url=True),
    ...         "tags": Field(css="span.tag::text", multi=True),
    ...     },
    ... )
    >>> result.record_count
    2
    >>> result.values[0]["name"]
    'First'
    >>> result.values[0]["link"]
    'https://example.org/p/1'
    >>> result.values[0]["tags"]
    ['a', 'b']
    >>> result.field_diagnostics["tags"].total_hits
    2
    >>> result.field_diagnostics["tags"].missing_records
    (1,)
    """
    if (record_xpath is None) == (record_css is None):
        raise FieldDefinitionError(
            "extract_records requires exactly one of the record_xpath "
            f"and record_css arguments, got record_xpath={record_xpath!r}, "
            f"record_css={record_css!r}"
        )
    if not fields:
        raise FieldDefinitionError("extract_records requires at least one field")

    merged_namespaces = _build_namespaces(selector, namespaces)
    resolved_base_url = _resolve_base_url(selector, base_url)
    records = _select_records(selector, record_xpath, record_css, merged_namespaces)
    extracted, hit_counts = _extract_all_records(
        records, fields, merged_namespaces, resolved_base_url
    )

    result = ExtractionResult(
        record_count=len(records),
        field_diagnostics=_build_diagnostics(tuple(fields), hit_counts),
        records=extracted,
    )
    required_missing = result.missing_required
    if required_missing:
        raise RecordExtractionError(result, required_missing)
    return result
