"""Guarded minimal DOCX text amendments using only the Python standard library.

Only nominated w:t nodes change. Other XML, runs, drawings, tables, equations,
relationships and package members retain their original bytes. Input is never
overwritten. An amendment JSON names the original SHA256 and exact paragraph IDs.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import posixpath
import re
import struct
import xml.etree.ElementTree as ET
import xml.parsers.expat
from xml.sax.saxutils import escape
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def paragraph_records(data):
    parser = xml.parsers.expat.ParserCreate()
    stack, active, records, texts = [], [], [], []

    def start(name, attributes):
        local = name.split(":")[-1]
        if stack:
            counts = stack[-1]["children"]
            counts[local] = counts.get(local, 0) + 1
            path = stack[-1]["path"] + f"/{local}[{counts[local]}]"
        else:
            path = f"/{local}[1]"
        stack.append({"local": local, "path": path, "children": {}})
        if name == "w:p":
            active.append({"id": path, "start": parser.CurrentByteIndex, "nodes": []})
        if name == "w:t" and active:
            index = parser.CurrentByteIndex
            end = data.index(b">", index) + 1
            texts.append({"open_start": index, "start": end, "pieces": [], "paragraph": active[-1],
                          "self_closing": data[index:end].rstrip().endswith(b"/>")})

    def characters(value):
        if texts and stack[-1]["local"] == "t":
            texts[-1]["pieces"].append(value)

    def end(name):
        if name == "w:t" and texts:
            node = texts.pop()
            node["end"] = parser.CurrentByteIndex
            node["text"] = "".join(node.pop("pieces"))
            owner = node.pop("paragraph")
            if not node.pop("self_closing"):
                owner["nodes"].append(node)
        if name == "w:p":
            record = active.pop()
            record["end"] = data.index(b">", parser.CurrentByteIndex) + 1
            record["text"] = "".join(n["text"] for n in record["nodes"])
            record["text_sha256"] = sha256(record["text"].encode())
            record["xml_sha256"] = sha256(data[record["start"]:record["end"]])
            records.append(record)
        stack.pop()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = characters
    parser.Parse(data, True)
    return sorted(records, key=lambda r: r["start"])


def index_document(path):
    path = Path(path)
    with zipfile.ZipFile(path) as package:
        bad = package.testzip()
        if bad:
            raise ValueError(f"Invalid source ZIP CRC: {bad}")
        document = package.read("word/document.xml")
        records = paragraph_records(document)
        members = {n: sha256(package.read(n)) for n in package.namelist()}
    return {"source_path": str(path.resolve()), "source_sha256": sha256(path.read_bytes()),
            "members_sha256": members,
            "paragraphs": [{k: v for k, v in r.items() if k not in ["nodes", "start", "end"]} for r in records]}, document, records


def distribute_text(nodes, replacement):
    """Preserve existing run placement for equal text; anchor new text at the edit."""
    old = "".join(n["text"] for n in nodes)
    spans = []
    offset = 0
    for node in nodes:
        spans.append((offset, offset + len(node["text"])))
        offset += len(node["text"])
    output = [""] * len(nodes)
    for operation, a, b, c, d in difflib.SequenceMatcher(None, old, replacement, autojunk=False).get_opcodes():
        if operation == "equal":
            for i, (lo, hi) in enumerate(spans):
                x, y = max(lo, a), min(hi, b)
                if y > x:
                    output[i] += old[x:y]
        elif operation in ["insert", "replace"]:
            candidates = [i for i, (lo, hi) in enumerate(spans) if lo <= a < hi]
            index = candidates[0] if candidates else len(nodes) - 1
            output[index] += replacement[c:d]
    if "".join(output) != replacement:
        raise AssertionError("Run-preserving text distribution mismatch")
    return output


def resize_requested_drawings(document, relationships, media, requests):
    """Alter only nominated drawing heights; preserve width and other drawing bytes."""
    if not requests:
        return document, []
    targets = {r.attrib["Id"]: posixpath.normpath(posixpath.join("word", r.attrib["Target"]))
               for r in ET.fromstring(relationships)}
    drawings = list(re.finditer(rb"<w:drawing\b[^>]*>.*?</w:drawing>", document, re.S))
    edits, audit, seen = [], [], set()
    for item in requests:
        rid = item["relationship_id"]
        if rid in seen or targets.get(rid) != item["package_member"]:
            raise ValueError("Ambiguous or mismatched nominated drawing relationship")
        seen.add(rid)
        matches = [m for m in drawings if re.search(rb'\br:embed="' + re.escape(rid.encode()) + rb'"', m.group())]
        if len(matches) != 1:
            raise ValueError("Nominated drawing relationship must occur exactly once")
        match = matches[0]
        fragment = match.group()
        if sha256(fragment) != item["expected_drawing_sha256"]:
            raise ValueError("Nominated drawing SHA256 guard failed")
        if item.get("preserve_width_and_match_png_aspect") is not True:
            raise ValueError("Explicit preserve-width PNG aspect policy required")
        payload = media.get(item["package_member"], b"")
        if not payload.startswith(b"\x89PNG\r\n\x1a\n") or payload[12:16] != b"IHDR":
            raise ValueError("Drawing resize requires nominated replacement PNG dimensions")
        pixel_width, pixel_height = struct.unpack(">II", payload[16:24])
        if pixel_width <= 0 or pixel_height <= 0:
            raise ValueError("Invalid replacement PNG dimensions")
        extents = list(re.finditer(rb"<wp:extent\b[^>]*/>", fragment))
        transforms = list(re.finditer(rb"<a:xfrm\b[^>]*>.*?</a:xfrm>", fragment, re.S))
        if len(extents) != 1 or len(transforms) != 1:
            raise ValueError("Drawing must have one wp extent and one matching transform")
        inner = list(re.finditer(rb"<a:ext\b[^>]*/>", transforms[0].group()))
        if len(inner) != 1:
            raise ValueError("Drawing transform extent is ambiguous")
        locations = [(extents[0].start(), extents[0].end()),
                     (transforms[0].start() + inner[0].start(), transforms[0].start() + inner[0].end())]
        width, old_height = int(item["expected_width_emu"]), int(item["expected_height_emu"])
        new_height = round(width * pixel_height / pixel_width)
        changed = fragment
        for start, end in sorted(locations, reverse=True):
            tag = fragment[start:end]
            cx = re.search(rb'\bcx="(\d+)"', tag)
            cy = re.search(rb'\bcy="(\d+)"', tag)
            if cx is None or cy is None or int(cx.group(1)) != width or int(cy.group(1)) != old_height:
                raise ValueError("Drawing extent width/height guard failed")
            tag = tag[:cy.start(1)] + str(new_height).encode() + tag[cy.end(1):]
            changed = changed[:start] + tag + changed[end:]
        edits.append((match.start(), match.end(), changed))
        audit.append({"relationship_id": rid, "package_member": item["package_member"],
                      "old_drawing_sha256": sha256(fragment), "new_drawing_sha256": sha256(changed),
                      "width_emu_preserved": width, "old_height_emu": old_height, "new_height_emu": new_height,
                      "png_dimensions": [pixel_width, pixel_height], "extent_pairs_changed": 2})
    result = document
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    after = list(re.finditer(rb"<w:drawing\b[^>]*>.*?</w:drawing>", result, re.S))
    if len(after) != len(drawings):
        raise AssertionError("Drawing count changed")
    for old, new in zip(drawings, after):
        if not any(old.start() == e[0] for e in edits) and old.group() != new.group():
            raise AssertionError("Unrequested drawing bytes changed")
    return result, audit


def apply_amendments(source, amendments_path, output, audit_path):
    source, output, audit_path = Path(source), Path(output), Path(audit_path)
    if source.resolve() == output.resolve():
        raise ValueError("Source must not be overwritten")
    if output.exists():
        raise FileExistsError(f"Versioned derivative already exists: {output}")
    manifest, document, records = index_document(source)
    requested = json.loads(Path(amendments_path).read_text(encoding="utf-8"))
    if requested.get("ready_to_apply") is not True:
        raise ValueError("Amendment set is a preparation draft; endpoint evidence has not marked it ready_to_apply")
    if requested["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Source SHA256 differs from reviewed authority")
    indexed = {r["id"]: r for r in records}
    edits, changed = [], []
    seen = set()
    for item in requested["amendments"]:
        key = item["id"]
        if key in seen:
            raise ValueError(f"Duplicate amendment: {key}")
        seen.add(key)
        record = indexed[key]
        if item["expected_text_sha256"] != record["text_sha256"]:
            raise ValueError(f"Paragraph text guard failed: {key}")
        fragment = document[record["start"]:record["end"]]
        if b"<w:fldChar" in fragment or b"<m:oMath" in fragment or b"<w:del" in fragment:
            raise ValueError(f"Do not alter field/equation/tracked-deletion paragraphs with this helper: {key}")
        replacement = item["replacement_text"]
        if not isinstance(replacement, str) or any(t in replacement for t in ["{{", "[INSERT", "[TODO", "<<"]):
            raise ValueError(f"Unresolved amendment text: {key}")
        if not record["nodes"]:
            raise ValueError(f"Paragraph has no editable text nodes: {key}")
        values = distribute_text(record["nodes"], replacement)
        for node, value in zip(record["nodes"], values):
            if value != node["text"]:
                edits.append((node["start"], node["end"], escape(value).encode("utf-8")))
                opening = document[node["open_start"]:node["start"]]
                if value and (value[0].isspace() or value[-1].isspace()) and b"xml:space=" not in opening:
                    edits.append((node["open_start"], node["start"], opening[:-1] + b' xml:space="preserve">'))
        appended = item.get("append_paragraphs", [])
        if appended:
            if not re.fullmatch(r"/document\[1\]/body\[1\]/p\[\d+\]", key):
                raise ValueError("Only explicit body-paragraph insertions are supported")
            style = re.search(rb"<w:pPr(?:\s[^>]*)?>.*?</w:pPr>", fragment, re.S)
            paragraph_style = style.group() if style else b""
            new_paragraphs = []
            for text in appended:
                if not isinstance(text, str) or any(t in text for t in ["{{", "[INSERT", "[TODO", "<<"]):
                    raise ValueError("Unresolved inserted paragraph")
                new_paragraphs.append(b"<w:p>" + paragraph_style + b'<w:r><w:t xml:space="preserve">' + escape(text).encode("utf-8") + b"</w:t></w:r></w:p>")
            edits.append((record["end"], record["end"], b"".join(new_paragraphs)))
        changed.append({"id": key, "before": record["text"], "after": replacement,
                        "appended_paragraphs": appended,
                        "reason": item.get("reason", ""), "evidence": item.get("evidence", [])})
    updated = document
    for a, b, text in sorted(edits, reverse=True):
        updated = updated[:a] + text + updated[b:]
    updated_records = paragraph_records(updated)
    n_added = sum(len(r["appended_paragraphs"]) for r in changed)
    if not n_added and [r["id"] for r in updated_records] != [r["id"] for r in records]:
        raise AssertionError("Paragraph/table structure changed")
    changes = {r["id"]: r for r in changed}
    expected_text = []
    for record in records:
        change = changes.get(record["id"])
        expected_text.append(change["after"] if change else record["text"])
        if change:
            expected_text.extend(change["appended_paragraphs"])
    if [r["text"] for r in updated_records] != expected_text:
        raise AssertionError("Unrequested paragraph text changed")
    replacements = {}
    for item in requested.get("media_replacements", []):
        member = item["package_member"]
        if member in replacements or not member.startswith("word/media/") or member not in manifest["members_sha256"]:
            raise ValueError(f"Invalid or repeated explicit media replacement: {member}")
        if item["expected_sha256"] != manifest["members_sha256"][member]:
            raise ValueError(f"Original media hash guard failed: {member}")
        payload = Path(item["replacement_path"]).read_bytes()
        if sha256(payload) != item["replacement_sha256"]:
            raise ValueError(f"New figure hash guard failed: {member}")
        if member.endswith(".png") and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("PNG replacement member requires PNG artwork")
        replacements[member] = payload
    with zipfile.ZipFile(source) as original:
        updated, drawing_audit = resize_requested_drawings(
            updated, original.read("word/_rels/document.xml.rels"), replacements,
            requested.get("drawing_resizes", []))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output, "w") as derivative:
        for info in original.infolist():
            derivative.writestr(info, updated if info.filename == "word/document.xml" else replacements.get(info.filename, original.read(info.filename)))
    final_manifest, _, _ = index_document(output)
    unchanged = [n for n in manifest["members_sha256"] if n != "word/document.xml" and n not in replacements]
    if any(manifest["members_sha256"][n] != final_manifest["members_sha256"][n] for n in unchanged):
        raise AssertionError("Protected package member changed")
    audit = {"status": "PASS_PACKAGE_TEXT_CHECKS_RENDER_PENDING", "source_sha256": manifest["source_sha256"],
             "output_sha256": final_manifest["source_sha256"], "source_path": manifest["source_path"],
             "output_path": str(output.resolve()), "amendments": changed,
             "unchanged_package_members": len(unchanged),
             "unchanged_media_members": len([n for n in unchanged if n.startswith("word/media/")]),
             "explicit_media_replacements": requested.get("media_replacements", []),
             "explicit_drawing_layout_amendments": drawing_audit,
             "explicit_paragraphs_inserted": n_added,
             "original_paragraphs_and_tables_preserved": True, "zip_crc_valid": True,
             "render_status": "NOT_RUN"}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--source", required=True)
    index.add_argument("--output", required=True)
    apply = sub.add_parser("apply")
    for field in ["source", "amendments", "output", "audit"]:
        apply.add_argument("--" + field, required=True)
    args = parser.parse_args()
    if args.command == "index":
        manifest, _, _ = index_document(args.source)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Indexed {len(manifest['paragraphs'])} paragraphs; original SHA256 {manifest['source_sha256']}")
    else:
        audit = apply_amendments(args.source, args.amendments, args.output, args.audit)
        print(f"Amended {len(audit['amendments'])} paragraphs; preserved {audit['unchanged_media_members']} media objects. Rendering still required.")


if __name__ == "__main__":
    main()
