import importlib.util
import json
from pathlib import Path
import struct
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/update_manuscript_docx.py"
spec = importlib.util.spec_from_file_location("update_manuscript", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "original.docx"
    document = b'''<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:spacing w:after="60"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>29</w:t></w:r><w:r><w:t>1 profiles</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Untouched table</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'''
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", document)
        z.writestr("word/media/image1.png", b"protected-original-artwork")
        z.writestr("word/_rels/document.xml.rels", b"protected-relationships")
    return path


def amendment(source, tmp_path, replacement):
    manifest, _, _ = module.index_document(source)
    row = manifest["paragraphs"][0]
    data = {"ready_to_apply": True, "source_sha256": manifest["source_sha256"], "amendments": [{"id": row["id"], "expected_text_sha256": row["text_sha256"], "replacement_text": replacement}]}
    path = tmp_path / "amendments.json"
    path.write_text(json.dumps(data))
    return path


def test_minimal_cross_run_amendment_preserves_structure_media_and_source(source, tmp_path):
    before = source.read_bytes()
    updates = amendment(source, tmp_path, "297 recorded profiles")
    output = tmp_path / "corrected.docx"
    report = module.apply_amendments(source, updates, output, tmp_path / "audit.json")
    assert source.read_bytes() == before
    manifest, document, _ = module.index_document(output)
    assert [r["text"] for r in manifest["paragraphs"]] == ["297 recorded profiles", "Untouched table"]
    assert b"<w:rPr><w:b/></w:rPr>" in document
    assert b'<w:spacing w:after="60"/>' in document
    assert report["unchanged_media_members"] == 1
    assert report["zip_crc_valid"]


def test_source_and_paragraph_guards_fail_without_output(source, tmp_path):
    updates = amendment(source, tmp_path, "297 profiles")
    data = json.loads(updates.read_text())
    data["amendments"][0]["expected_text_sha256"] = "incorrect"
    updates.write_text(json.dumps(data))
    output = tmp_path / "must_not_exist.docx"
    with pytest.raises(ValueError, match="text guard"):
        module.apply_amendments(source, updates, output, tmp_path / "audit.json")
    assert not output.exists()
    data["source_sha256"] = "incorrect"
    updates.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Source SHA256"):
        module.apply_amendments(source, updates, output, tmp_path / "audit.json")


def test_never_overwrites_source_or_exports_unresolved_placeholder(source, tmp_path):
    updates = amendment(source, tmp_path, "[INSERT count] profiles")
    with pytest.raises(ValueError, match="Source must not be overwritten"):
        module.apply_amendments(source, updates, source, tmp_path / "audit.json")
    with pytest.raises(ValueError, match="Unresolved amendment"):
        module.apply_amendments(source, updates, tmp_path / "corrected.docx", tmp_path / "audit.json")


def test_explicit_insertions_and_allowlisted_artwork_are_guarded(source, tmp_path):
    updates = amendment(source, tmp_path, "297 profiles")
    data = json.loads(updates.read_text())
    data["amendments"][0]["append_paragraphs"] = ["An independent evaluation was specified."]
    replacement = tmp_path / "new.png"
    replacement.write_bytes(b"\x89PNG\r\n\x1a\nnew synthetic artwork")
    manifest, _, _ = module.index_document(source)
    data["media_replacements"] = [{"package_member": "word/media/image1.png", "expected_sha256": manifest["members_sha256"]["word/media/image1.png"], "replacement_path": str(replacement), "replacement_sha256": module.sha256(replacement.read_bytes())}]
    updates.write_text(json.dumps(data))
    output = tmp_path / "corrected.docx"
    report = module.apply_amendments(source, updates, output, tmp_path / "audit.json")
    indexed, _, _ = module.index_document(output)
    assert [r["text"] for r in indexed["paragraphs"]] == ["297 profiles", "An independent evaluation was specified.", "Untouched table"]
    assert report["explicit_paragraphs_inserted"] == 1
    with zipfile.ZipFile(output) as z:
        assert z.read("word/media/image1.png") == replacement.read_bytes()
        assert z.read("word/_rels/document.xml.rels") == b"protected-relationships"


def test_explicit_drawing_resize_preserves_width_aspect_and_other_drawings():
    first = b'<w:drawing><wp:inline><wp:extent cx="6217920" cy="3500000"/><a:graphic><a:blip r:embed="rId16"/><a:xfrm><a:off x="0" y="0"/><a:ext cx="6217920" cy="3500000"/></a:xfrm></a:graphic></wp:inline></w:drawing>'
    other = first.replace(b"rId16", b"rId17")
    document = b"before" + first + b"middle" + other + b"after"
    relationships = b'<Relationships><Relationship Id="rId16" Target="media/image11.png"/><Relationship Id="rId17" Target="media/image12.png"/></Relationships>'
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 820, 670)
    request = {"relationship_id": "rId16", "package_member": "word/media/image11.png",
               "expected_drawing_sha256": module.sha256(first), "expected_width_emu": 6217920,
               "expected_height_emu": 3500000, "preserve_width_and_match_png_aspect": True}
    result, audit = module.resize_requested_drawings(document, relationships, {"word/media/image11.png": png}, [request])
    height = round(6217920 * 670 / 820)
    assert result == b"before" + first.replace(b'cy="3500000"', f'cy="{height}"'.encode()) + b"middle" + other + b"after"
    assert audit[0]["width_emu_preserved"] == 6217920 and audit[0]["extent_pairs_changed"] == 2
    for field, value, error in [("expected_width_emu", 1, "width/height guard"),
                                 ("expected_drawing_sha256", "wrong", "SHA256 guard"),
                                 ("package_member", "word/media/other.png", "relationship")]:
        with pytest.raises(ValueError, match=error):
            module.resize_requested_drawings(document, relationships, {"word/media/image11.png": png}, [{**request, field: value}])
