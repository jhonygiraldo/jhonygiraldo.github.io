#!/usr/bin/env python3
"""Validate structured content and render reusable Quarto include fragments."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "_data"
OUTPUT_DIR = ROOT / "_includes" / "generated"
BIB_PATH = ROOT / "_bibliography" / "papers.bib"


def load_json(name: str) -> Any:
    path = DATA_DIR / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path.relative_to(ROOT)}: {exc}") from exc


def parse_bibtex(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    entry_start = re.compile(r"@(\w+)\s*\{\s*([^,]+)\s*,", re.MULTILINE)
    position = 0

    while True:
        match = entry_start.search(text, position)
        if not match:
            break
        entry_type, key = match.group(1).lower(), match.group(2).strip()
        depth = 1
        cursor = match.end()
        quote = False
        escaped = False
        while cursor < len(text) and depth:
            character = text[cursor]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quote = not quote
            elif not quote and character == "{":
                depth += 1
            elif not quote and character == "}":
                depth -= 1
            cursor += 1

        if depth:
            raise ValueError(f"Unclosed BibTeX entry: {key}")

        if entry_type != "string":
            body = text[match.end() : cursor - 1]
            fields = parse_bibtex_fields(body)
            fields.update({"type": entry_type, "key": key})
            entries.append(fields)
        position = cursor

    return entries


def parse_bibtex_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    cursor = 0
    length = len(body)

    while cursor < length:
        while cursor < length and (body[cursor].isspace() or body[cursor] == ","):
            cursor += 1
        match = re.match(r"([A-Za-z_][\w-]*)\s*=\s*", body[cursor:])
        if not match:
            break
        name = match.group(1).lower()
        cursor += match.end()
        if cursor >= length:
            break

        if body[cursor] == "{":
            cursor += 1
            start = cursor
            depth = 1
            escaped = False
            while cursor < length and depth:
                character = body[cursor]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                cursor += 1
            value = body[start : cursor - 1]
        elif body[cursor] == '"':
            cursor += 1
            start = cursor
            escaped = False
            while cursor < length:
                character = body[cursor]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    break
                cursor += 1
            value = body[start:cursor]
            cursor += 1
        else:
            start = cursor
            while cursor < length and body[cursor] != ",":
                cursor += 1
            value = body[start:cursor]

        fields[name] = value.strip()

    return fields


LATEX_REPLACEMENTS = {
    r"\&": "&",
    r"\_": "_",
    r"\textendash": "–",
    r"\textemdash": "—",
    r"\i": "ı",
    "{\\'e}": "é",
    "{\\'E}": "É",
    "{\\'i}": "í",
    "{\\'a}": "á",
    "{\\'o}": "ó",
    "{\\'u}": "ú",
    "{\\~n}": "ñ",
    "{\\~N}": "Ñ",
    "{\\c{c}}": "ç",
}


def latex_to_text(value: str) -> str:
    output = value
    for source, replacement in LATEX_REPLACEMENTS.items():
        output = output.replace(source, replacement)
    output = re.sub(r"\{\\['`^\"~]([A-Za-zı])\}", r"\1", output)
    output = re.sub(r"\\(?:textit|emph|textbf)\{([^{}]+)\}", r"\1", output)
    output = output.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", output).strip()


def format_author(author: str) -> str:
    author = latex_to_text(author.strip())
    if author.count(",") == 1:
        last, first = (part.strip() for part in author.split(",", 1))
        author = f"{first} {last}"
    escaped = html.escape(author)
    if re.search(r"\bJhony\s+(?:H\.\s+)?Giraldo\b", author, re.IGNORECASE):
        return f"<strong>{escaped}</strong>"
    return escaped


def author_list(value: str) -> str:
    authors = [format_author(author) for author in re.split(r"\s+and\s+", value) if author.strip()]
    if len(authors) < 2:
        return authors[0] if authors else ""
    if len(authors) == 2:
        return " and ".join(authors)
    return ", ".join(authors[:-1]) + ", and " + authors[-1]


def anchor(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def internal_url(value: str) -> str:
    if value.startswith(("https://", "http://", "mailto:", "/")):
        return value
    return "/" + value.lstrip("./")


def raw_block(lines: list[str]) -> list[str]:
    return ["```{=html}", *lines, "```"]


def validate(news: list[dict[str, Any]], people: dict[str, Any], courses: list[dict[str, Any]], publications: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []

    for index, item in enumerate(news, start=1):
        for field in ("date", "title", "body"):
            if not item.get(field):
                errors.append(f"news item {index} is missing {field}")
        try:
            date.fromisoformat(item.get("date", ""))
        except ValueError:
            errors.append(f"news item {index} has an invalid ISO date")

    current_names = [person.get("name", "") for person in people.get("current", [])]
    duplicate_current = [name for name, count in Counter(current_names).items() if name and count > 1]
    if duplicate_current:
        errors.append("duplicate current team members: " + ", ".join(duplicate_current))
    for group in ("current", "visitors", "alumni"):
        for index, person in enumerate(people.get(group, []), start=1):
            if not person.get("name"):
                errors.append(f"{group} person {index} is missing name")

    for index, course in enumerate(courses, start=1):
        for field in ("title", "level", "period", "summary"):
            if not course.get(field):
                errors.append(f"course {index} is missing {field}")

    keys = [publication.get("key", "") for publication in publications]
    duplicate_keys = [key for key, count in Counter(keys).items() if key and count > 1]
    if duplicate_keys:
        errors.append("duplicate BibTeX keys: " + ", ".join(duplicate_keys))
    for publication in publications:
        for field in ("title", "author", "year"):
            if not publication.get(field):
                errors.append(f"publication {publication.get('key', '?')} is missing {field}")
        if publication.get("author") and " and " not in publication["author"] and "," in publication["author"]:
            errors.append(f"publication {publication.get('key', '?')} may use commas instead of 'and' between authors")

    publication_keys = set(keys)
    for item in news:
        key = item.get("publication_key")
        if key and key not in publication_keys:
            errors.append(f"news item '{item.get('title')}' references unknown publication key {key}")

    return errors


def render_person_card(person: dict[str, Any]) -> str:
    name = html.escape(person["name"])
    if person.get("url"):
        name = f'<a href="{html.escape(person["url"], quote=True)}">{name}</a>'
    return "\n".join(
        [
            '<article class="person-card">',
            '  <div class="person-body">',
            f"    <h3>{name}</h3>",
            f'    <p class="person-role">{html.escape(person.get("role", ""))}</p>',
            f'    <p>{html.escape(person.get("topic", ""))}</p>',
            f'    <p class="person-links">{html.escape(person.get("institution", ""))}</p>',
            f'    <p class="person-dates">{html.escape(person.get("dates", ""))}</p>',
            "  </div>",
            "</article>",
        ]
    )


def render_people(people: dict[str, Any]) -> tuple[str, str]:
    current = people.get("current", [])
    visitors = people.get("visitors", [])
    alumni = people.get("alumni", [])

    full: list[str] = ["## Current team", ""]
    current_html = ['<div class="team-grid">']
    current_html.extend(render_person_card(person) for person in current)
    current_html.append("</div>")
    full.extend(raw_block(current_html))
    full.extend(["", "## Former visiting researchers", ""])
    visitors_html = ['<div class="team-grid">']
    visitors_html.extend(render_person_card(person) for person in visitors)
    visitors_html.append("</div>")
    full.extend(raw_block(visitors_html))
    full.extend(["", "## Alumni", ""])
    alumni_html = ['<ul class="alumni-list">']
    for person in alumni:
        name = html.escape(person["name"])
        if person.get("url"):
            name = f'<a href="{html.escape(person["url"], quote=True)}">{name}</a>'
        description = f'{html.escape(person.get("former_role", ""))}, {html.escape(person.get("dates", ""))}'
        if person.get("destination"):
            description += f' — now {html.escape(person["destination"])}'
        alumni_html.append(f"  <li><strong>{name}</strong> — {description}</li>")
    alumni_html.append("</ul>")
    full.extend(raw_block(alumni_html))
    full.extend(["", "## Master’s students", "", ", ".join(html.escape(name) for name in people.get("masters", [])) + "."])

    preview: list[str] = ['<div class="team-grid">']
    preview.extend(render_person_card(person) for person in current[:3])
    preview.append("</div>")
    return "\n".join(full) + "\n", "\n".join(raw_block(preview)) + "\n"


def render_courses(courses: list[dict[str, Any]]) -> str:
    lines = ['<div class="course-grid">']
    for course in courses:
        title = html.escape(course["title"])
        if course.get("url"):
            title = f'<a href="{internal_url(course["url"])}">{title}</a>'
        metadata = " · ".join(
            part for part in [course.get("level"), course.get("role"), course.get("period")] if part
        )
        lines.extend(
            [
                '<article class="course-card">',
                f'  <div class="course-code">{html.escape(course.get("code") or course.get("institution", ""))}</div>',
                f"  <h3>{title}</h3>",
                f"  <p>{html.escape(course['summary'])}</p>",
                f'  <p class="person-links">{html.escape(metadata)}</p>',
                "</article>",
            ]
        )
    lines.append("</div>")
    return "\n".join(raw_block(lines)) + "\n"


def news_link(item: dict[str, Any]) -> str | None:
    if item.get("publication_key"):
        return "/publications/#" + anchor(item["publication_key"])
    if item.get("url"):
        return internal_url(item["url"])
    return None


def render_news_card(item: dict[str, Any]) -> str:
    title = html.escape(item["title"])
    link = news_link(item)
    if link:
        title = f'<a href="{html.escape(link, quote=True)}">{title}</a>'
    formatted_date = date.fromisoformat(item["date"]).strftime("%d %b %Y")
    return "\n".join(
        [
            '<article class="news-card">',
            f'  <div class="news-date"><time datetime="{item["date"]}">{formatted_date}</time></div>',
            f"  <h3>{title}</h3>",
            f"  <p>{html.escape(item['body'])}</p>",
            "</article>",
        ]
    )


def render_news(news: list[dict[str, Any]]) -> tuple[str, str]:
    ordered = sorted(news, key=lambda item: item["date"], reverse=True)
    latest_html = ['<div class="news-grid">'] + [render_news_card(item) for item in ordered[:3]] + ["</div>"]
    latest = raw_block(latest_html)
    full: list[str] = []
    for year in sorted({item["date"][:4] for item in ordered}, reverse=True):
        year_html = ['<div class="news-grid">']
        year_html.extend(render_news_card(item) for item in ordered if item["date"].startswith(year))
        year_html.append("</div>")
        full.extend([f"## {year}", "", *raw_block(year_html), ""])
    return "\n".join(full), "\n".join(latest) + "\n"


def publication_links(publication: dict[str, str]) -> str:
    links: list[str] = []
    if publication.get("doi"):
        links.append(f'<a href="https://doi.org/{html.escape(publication["doi"], quote=True)}">DOI</a>')
    if publication.get("arxiv"):
        links.append(f'<a href="https://arxiv.org/abs/{html.escape(publication["arxiv"], quote=True)}">arXiv</a>')
    for field, label in (("html", "Project"), ("code", "Code"), ("pdf", "PDF"), ("website", "Website")):
        if publication.get(field):
            url = publication[field]
            if field == "pdf" and "://" not in url and not url.startswith("/"):
                url = "/assets/pdf/" + url
            links.append(f'<a href="{html.escape(url, quote=True)}">{label}</a>')
    return " ".join(links)


def render_publication(publication: dict[str, str]) -> str:
    title = html.escape(latex_to_text(publication["title"]))
    authors = author_list(publication["author"])
    venue = latex_to_text(publication.get("abbr") or publication.get("journal") or publication.get("booktitle") or publication["type"])
    periodical = latex_to_text(publication.get("journal") or publication.get("booktitle") or "")
    links = publication_links(publication)
    return "\n".join(
        [
            f'<article id="{anchor(publication["key"])}" class="publication-item" data-publication-item data-year="{publication["year"]}">',
            f'  <div class="publication-venue">{html.escape(venue)}</div>',
            "  <div>",
            f'    <div class="publication-title">{title}</div>',
            f'    <div class="publication-authors">{authors}</div>',
            f'    <div class="publication-meta">{html.escape(periodical)} · {html.escape(publication["year"])}</div>',
            f'    <div class="publication-links">{links}</div>' if links else "",
            "  </div>",
            "</article>",
        ]
    )


def render_publications(publications: list[dict[str, str]]) -> tuple[str, str]:
    # Python's sort is stable, so entries from the same year retain the
    # deliberately curated order used in the bibliography file.
    ordered = sorted(publications, key=lambda item: int(item.get("year", "0")), reverse=True)
    full = [
        '<div class="publication-tools">',
        '  <label><span class="visually-hidden">Search publications</span><input class="publication-search" type="search" placeholder="Search by title, author, venue, or year" data-publication-search></label>',
        f'  <div class="publication-count" data-publication-count>{len(ordered)} publications</div>',
        "</div>",
    ]
    by_year: dict[str, list[dict[str, str]]] = defaultdict(list)
    for publication in ordered:
        by_year[publication["year"]].append(publication)
    for year in sorted(by_year, reverse=True):
        full.append(f'<h2 class="publication-year" data-publication-year="{year}">{year}</h2>')
        full.extend(render_publication(publication) for publication in by_year[year])

    selected = [publication for publication in ordered if publication.get("selected", "").lower() == "true"][:6]
    selected_lines = [render_publication(publication) for publication in selected]
    selected_output = "\n".join(raw_block(selected_lines)) + "\n"
    return "\n".join(raw_block(full)) + "\n", selected_output


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate without writing generated includes")
    args = parser.parse_args()

    try:
        news = load_json("news.json")
        people = load_json("people.json")
        courses = load_json("courses.json")
        publications = parse_bibtex(BIB_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    errors = validate(news, people, courses, publications)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    if args.check:
        print(f"Validated {len(publications)} publications, {len(news)} news items, {len(people.get('current', []))} current team members, and {len(courses)} courses.")
        return 0

    people_full, people_preview = render_people(people)
    news_full, news_latest = render_news(news)
    publications_full, publications_selected = render_publications(publications)
    generated = {
        "people.md": people_full,
        "team-preview.md": people_preview,
        "courses.md": render_courses(courses),
        "news.md": news_full,
        "latest-news.md": news_latest,
        "publications.md": publications_full,
        "selected-publications.md": publications_selected,
    }
    for name, content in generated.items():
        write_if_changed(OUTPUT_DIR / name, content)
    print(f"Rendered {len(generated)} include files from structured content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
