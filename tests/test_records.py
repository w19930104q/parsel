"""Tests for one-call record extraction (:mod:`parsel.records`)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest
from cssselect import SelectorError

import parsel
from parsel import (
    ExtractionResult,
    Field,
    FieldDefinitionError,
    FieldDiagnostics,
    Record,
    RecordExtractionError,
    Selector,
    extract_records,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

HTML_BODY = """
<html><body>
<ul class="products">
  <li class="product">
    <a href="/product/1">First product</a>
    <span class="price">10</span>
    <span class="tag">red</span>
    <span class="tag">small</span>
  </li>
  <li class="product">
    <a href="/product/2">Second product</a>
    <span class="price">20</span>
  </li>
  <li class="product">
    <span class="price">30</span>
  </li>
</ul>
</body></html>
"""

XML_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://example.com/catalog">
  <item>
    <name>First item</name>
    <link href="/item/1"/>
    <category>tools</category>
    <category>garden</category>
  </item>
  <item>
    <name>Second item</name>
    <link href="https://cdn.example.net/item/2"/>
  </item>
</catalog>
"""


def _fields() -> Mapping[str, Field]:
    return {
        "name": Field(css="a::text"),
        "link": Field(css="a", attr="href", url=True),
        "price": Field(xpath="./span[@class='price']/text()"),
        "tags": Field(css="span.tag::text", multi=True),
    }


class TestExtractRecordsHTML:
    def test_repeated_records_and_per_record_values(self) -> None:
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        result = sel.extract_records(record_css="li.product", fields=_fields())

        assert isinstance(result, ExtractionResult)
        assert result.record_count == 3
        assert all(isinstance(record, Record) for record in result.records)
        assert [record.index for record in result.records] == [0, 1, 2]
        assert result.values == (
            {
                "name": "First product",
                "link": "https://shop.example.com/product/1",
                "price": "10",
                "tags": ["red", "small"],
            },
            {
                "name": "Second product",
                "link": "https://shop.example.com/product/2",
                "price": "20",
                "tags": [],
            },
            {
                "name": None,
                "link": None,
                "price": "30",
                "tags": [],
            },
        )

    def test_record_xpath_equivalent_to_record_css(self) -> None:
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        result = extract_records(
            sel,
            record_xpath="//li[@class='product']",
            fields=_fields(),
        )
        assert result.record_count == 3
        assert result.values[0]["name"] == "First product"

    def test_field_xpath_and_attribute(self) -> None:
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        result = extract_records(
            sel,
            record_css="li.product",
            fields={
                "name": Field(xpath=".//a/text()"),
                "link_xpath_attr": Field(xpath=".//a/@href", url=True),
                "link_css_attr": Field(css="a", attr="href", url=True),
            },
        )
        first = result.values[0]
        assert first["link_xpath_attr"] == "https://shop.example.com/product/1"
        assert first["link_css_attr"] == first["link_xpath_attr"]

    def test_multi_value_vs_single_value(self) -> None:
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        single = extract_records(
            sel,
            record_css="li.product",
            fields={"tag": Field(css="span.tag::text")},
        )
        multiple = extract_records(
            sel,
            record_css="li.product",
            fields={"tags": Field(css="span.tag::text", multi=True)},
        )
        assert single.values[0]["tag"] == "red"
        assert multiple.values[0]["tags"] == ["red", "small"]
        # Optional multi field without hits is an empty list, not None.
        assert multiple.values[1]["tags"] == []
        # Single optional field without hits is None.
        assert single.values[1]["tag"] is None

    def test_diagnostics_hit_counts_and_missing(self) -> None:
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        result = sel.extract_records(record_css="li.product", fields=_fields())
        diagnostics = result.field_diagnostics

        assert isinstance(diagnostics["name"], FieldDiagnostics)
        assert diagnostics["name"].total_hits == 2
        assert diagnostics["name"].hit_records == (0, 1)
        assert diagnostics["name"].missing_records == (2,)
        assert diagnostics["name"].hit_count == 2

        assert diagnostics["tags"].total_hits == 2
        assert diagnostics["tags"].hit_records == (0,)
        assert diagnostics["tags"].missing_records == (1, 2)

        assert diagnostics["price"].total_hits == 3
        assert diagnostics["price"].missing_records == ()

    def test_record_get_helper(self) -> None:
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        result = sel.extract_records(record_css="li.product", fields=_fields())
        assert result.records[2].get("price") == "30"
        assert result.records[2].get("name") is None
        # A field that was never declared is absent from the mapping.
        assert result.records[2].get("unexpected", "fallback") == "fallback"


class TestEmptyPage:
    def test_no_records_returns_empty_result(self) -> None:
        result = extract_records(
            Selector(text="<html><body></body></html>"),
            record_css="li.product",
            fields={"name": Field(css="a::text", required=True)},
        )
        assert result.record_count == 0
        assert result.records == ()
        assert result.values == ()
        assert result.missing_required == ()
        diagnostics = result.field_diagnostics["name"]
        assert diagnostics.total_hits == 0
        assert diagnostics.hit_records == ()
        assert diagnostics.missing_records == ()

    def test_empty_xml_document(self) -> None:
        result = extract_records(
            Selector(text="<?xml version='1.0'?><catalog/>", type="xml"),
            record_xpath="//item",
            fields={"name": Field(xpath="./name/text()")},
        )
        assert result.record_count == 0
        assert result.field_diagnostics["name"].total_hits == 0


class TestRequiredFields:
    def test_missing_required_raises_and_keeps_data(self) -> None:
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        with pytest.raises(RecordExtractionError) as exc_info:
            extract_records(
                sel,
                record_css="li.product",
                fields={"name": Field(css="a::text", required=True)},
            )
        error = exc_info.value
        assert error.missing == ((2, "name"),)
        assert "record 2" in str(error)
        assert "name" in str(error)
        # The partially extracted data remains available on the exception.
        assert error.result.record_count == 3
        assert error.result.records[0].values["name"] == "First product"
        assert error.result.records[2].missing_required == ("name",)
        assert error.result.missing_required == ((2, "name"),)

    def test_several_missing_required_fields_are_grouped(self) -> None:
        sel = Selector(text="<ul><li><b>x</b></li></ul>")
        with pytest.raises(RecordExtractionError) as exc_info:
            extract_records(
                sel,
                record_css="li",
                fields={
                    "first": Field(css="em::text", required=True),
                    "second": Field(css="u::text", required=True),
                },
            )
        error = exc_info.value
        assert error.missing == ((0, "first"), (0, "second"))
        assert str(error) == "Missing required field(s): record 0: first, second"

    def test_present_required_field_does_not_raise(self) -> None:
        sel = Selector(text="<ul><li><a>x</a></li></ul>")
        result = extract_records(
            sel,
            record_css="li",
            fields={"name": Field(css="a::text", required=True)},
        )
        assert result.values[0]["name"] == "x"
        assert result.missing_required == ()


class TestDefaultNamespace:
    def test_default_namespace_xpath(self) -> None:
        sel = Selector(text=XML_BODY, type="xml", base_url="https://feed.example/")
        result = extract_records(
            sel,
            record_xpath="//d:item",
            fields={
                "name": Field(xpath="./d:name/text()", required=True),
                "link": Field(xpath="./d:link", attr="href", url=True),
                "categories": Field(xpath="./d:category/text()", multi=True),
            },
        )
        assert result.record_count == 2
        assert result.values == (
            {
                "name": "First item",
                "link": "https://feed.example/item/1",
                "categories": ["tools", "garden"],
            },
            {
                "name": "Second item",
                "link": "https://cdn.example.net/item/2",
                "categories": [],
            },
        )

    def test_default_namespace_css(self) -> None:
        sel = Selector(text=XML_BODY, type="xml")
        result = extract_records(
            sel,
            record_css="d|item",
            fields={"name": Field(css="d|name::text", required=True)},
        )
        assert [value["name"] for value in result.values] == [
            "First item",
            "Second item",
        ]

    def test_explicit_namespace_mapping_override(self) -> None:
        sel = Selector(text=XML_BODY, type="xml")
        result = extract_records(
            sel,
            record_xpath="//atom:item",
            namespaces={"atom": "http://example.com/catalog"},
            fields={"name": Field(xpath="./atom:name/text()")},
        )
        assert result.values[0]["name"] == "First item"

    def test_no_namespace_document_needs_no_prefix(self) -> None:
        sel = Selector(
            text="<?xml version='1.0'?><catalog><item><name>x</name></item></catalog>",
            type="xml",
        )
        result = extract_records(
            sel,
            record_xpath="//item",
            fields={"name": Field(xpath="./name/text()", required=True)},
        )
        assert result.values[0]["name"] == "x"


class TestRelativeURLs:
    def test_relative_and_absolute_and_protocol_relative(self) -> None:
        body = """
        <ul>
          <li><a href="page/1">a</a></li>
          <li><a href="https://other.example/x">b</a></li>
          <li><a href="//cdn.example/z">c</a></li>
          <li><a href="../top">d</a></li>
        </ul>
        """
        sel = Selector(text=body, base_url="https://site.example/root/index.html")
        result = extract_records(
            sel,
            record_css="li",
            fields={"url": Field(css="a", attr="href", url=True)},
        )
        assert [value["url"] for value in result.values] == [
            "https://site.example/root/page/1",
            "https://other.example/x",
            "https://cdn.example/z",
            "https://site.example/top",
        ]

    def test_multi_url_field_normalizes_every_value(self) -> None:
        body = '<ul><li><a href="a">1</a><a href="b">2</a></li></ul>'
        sel = Selector(text=body, base_url="https://site.example/root/")
        result = extract_records(
            sel,
            record_css="li",
            fields={"urls": Field(css="a", attr="href", url=True, multi=True)},
        )
        assert result.values[0]["urls"] == [
            "https://site.example/root/a",
            "https://site.example/root/b",
        ]

    def test_html_base_tag_is_used_as_fallback(self) -> None:
        body = (
            '<html><head><base href="https://doc.example/root/"></head>'
            "<body><ul><li><a href='p'>a</a></li></ul></body></html>"
        )
        result = extract_records(
            Selector(text=body),
            record_css="li",
            fields={"url": Field(css="a", attr="href", url=True)},
        )
        assert result.values[0]["url"] == "https://doc.example/root/p"

    def test_explicit_base_url_argument_wins(self) -> None:
        sel = Selector(text="<a href='x'>x</a>", base_url="https://old.example/")
        result = extract_records(
            sel,
            record_xpath="//a",
            base_url="https://new.example/",
            fields={"url": Field(xpath="./@href", url=True)},
        )
        assert result.values[0]["url"] == "https://new.example/x"

    def test_url_field_without_base_url_raises(self) -> None:
        sel = Selector(text="<ul><li><a href='/x'>a</a></li></ul>")
        with pytest.raises(ValueError, match="base URL"):
            extract_records(
                sel,
                record_css="li",
                fields={"url": Field(css="a", attr="href", url=True)},
            )

    def test_non_url_field_does_not_require_base_url(self) -> None:
        result = extract_records(
            Selector(text="<ul><li><a href='/x'>a</a></li></ul>"),
            record_css="li",
            fields={"href": Field(css="a", attr="href")},
        )
        assert result.values[0]["href"] == "/x"


class TestBadSelectorsAndArguments:
    def test_invalid_record_xpath_raises_value_error(self) -> None:
        sel = Selector(text=HTML_BODY)
        with pytest.raises(ValueError, match="XPath error"):
            extract_records(
                sel,
                record_xpath="//[",
                fields={"name": Field(css="a::text")},
            )

    def test_invalid_field_xpath_raises_value_error(self) -> None:
        sel = Selector(text=HTML_BODY)
        with pytest.raises(ValueError, match="XPath error"):
            extract_records(
                sel,
                record_css="li",
                fields={"name": Field(xpath="./[")},
            )

    def test_invalid_css_raises_selector_error(self) -> None:
        sel = Selector(text=HTML_BODY)
        with pytest.raises(SelectorError):
            extract_records(
                sel,
                record_css="???",
                fields={"name": Field(css="a::text")},
            )
        with pytest.raises(SelectorError):
            extract_records(
                sel,
                record_css="li",
                fields={"name": Field(css="???")},
            )

    def test_error_semantics_match_raw_selector_api(self) -> None:
        sel = Selector(text=HTML_BODY)
        with pytest.raises(ValueError, match="XPath error") as raw:
            sel.xpath("//[")
        with pytest.raises(ValueError, match="XPath error") as wrapped:
            extract_records(
                sel,
                record_xpath="//[",
                fields={"name": Field(xpath=".")},
            )
        assert str(raw.value) == str(wrapped.value)

    def test_field_requires_exactly_one_expression(self) -> None:
        with pytest.raises(FieldDefinitionError):
            Field(xpath="//a", css="a")
        with pytest.raises(FieldDefinitionError):
            Field()

    def test_record_requires_exactly_one_selector(self) -> None:
        sel = Selector(text=HTML_BODY)
        fields = {"name": Field(css="a::text")}
        with pytest.raises(FieldDefinitionError):
            extract_records(sel, fields=fields)
        with pytest.raises(FieldDefinitionError):
            extract_records(sel, fields=fields, record_xpath="//li", record_css="li")

    def test_empty_field_mapping_rejected(self) -> None:
        sel = Selector(text=HTML_BODY)
        with pytest.raises(FieldDefinitionError, match="at least one field"):
            extract_records(sel, record_css="li", fields={})

    def test_empty_attr_rejected(self) -> None:
        with pytest.raises(FieldDefinitionError):
            Field(css="a", attr="")

    def test_frozen_field_and_records(self) -> None:
        field = Field(css="a::text")
        with pytest.raises(FrozenInstanceError):
            field.required = True  # type: ignore[misc]
        sel = Selector(text=HTML_BODY, base_url="https://shop.example.com/")
        result = sel.extract_records(record_css="li.product", fields=_fields())
        with pytest.raises(FrozenInstanceError):
            result.records[0].index = 5  # type: ignore[misc]


class TestExistingAPIUnchanged:
    def test_selector_and_selector_list_api_still_work(self) -> None:
        sel = Selector(text=HTML_BODY)
        assert sel.css("li.product").xpath(".//a/text()").getall() == [
            "First product",
            "Second product",
        ]
        assert sel.xpath("//span[@class='price']/text()").getall() == [
            "10",
            "20",
            "30",
        ]
        assert isinstance(sel.css("li.product").getall(), list)
        assert sel.css("li.missing").get() is None

    def test_xpath_unsupported_on_json_selector(self) -> None:
        sel = Selector(text='{"items": []}')
        with pytest.raises(ValueError, match="Cannot use xpath"):
            sel.xpath("//item")
        with pytest.raises(ValueError, match="Cannot use xpath"):
            extract_records(
                sel,
                record_xpath="//item",
                fields={"name": Field(xpath="./text()")},
            )


def test_result_types_are_public() -> None:
    public_names = (
        "ExtractionResult",
        "Field",
        "FieldDefinitionError",
        "FieldDiagnostics",
        "Record",
        "RecordExtractionError",
        "extract_records",
    )
    for name in public_names:
        assert hasattr(parsel, name)
