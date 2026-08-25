from __future__ import annotations

import base64
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
import zlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "check_local_links.py",
    "public_release_audit.py",
    "release_manifest.py",
    "release_tree.py",
)
MANIFEST_HEADER = "# sha256\\tbytes\\tgit_index_path; self excluded\n"


def _write_manifest(root: Path) -> None:
    relative_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "RELEASE_FILES.sha256"
    )
    rows = []
    for relative in relative_paths:
        raw = (root / relative).read_bytes()
        rows.append(f"{hashlib.sha256(raw).hexdigest()}\t{len(raw)}\t{relative}")
    (root / "RELEASE_FILES.sha256").write_text(
        MANIFEST_HEADER + "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run(
    root: Path,
    script: str,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_environment = os.environ.copy()
    if environment is not None:
        process_environment.update(environment)
    return subprocess.run(
        [sys.executable, "-B", f"scripts/{script}", *arguments],
        cwd=root,
        check=False,
        env=process_environment,
        text=True,
        capture_output=True,
    )


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )


def _png_with_text(text: bytes) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"tEXt", b"Description\x00" + text)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


def test_release_tools_bind_a_codeload_style_archive_without_vacuous_pass() -> None:
    # Nest under the real checkout to prove rev-parse cannot attach to a parent index.
    with tempfile.TemporaryDirectory(prefix=".phasepair-archive-", dir=REPOSITORY_ROOT) as raw:
        archive = Path(raw)
        scripts = archive / "scripts"
        scripts.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)
        (archive / "README.md").write_text("[status](STATUS.md)\n", encoding="utf-8")
        (archive / "STATUS.md").write_text("archive fixture\n", encoding="utf-8")
        _write_manifest(archive)

        manifest = _run(archive, "release_manifest.py", "verify")
        audit = _run(archive, "public_release_audit.py")
        links = _run(archive, "check_local_links.py")
        assert manifest.returncode == 0, manifest.stderr
        assert manifest.stdout.strip() == "RELEASE_MANIFEST_VERIFIED"
        assert audit.returncode == 0, audit.stdout + audit.stderr
        assert audit.stdout.strip().startswith("PUBLIC_RELEASE_AUDIT_PASS files=7 ")
        assert links.returncode == 0, links.stdout + links.stderr
        assert links.stdout.strip() == "LOCAL_MARKDOWN_LINK_AUDIT_PASS links=1"
        assert not list(archive.rglob("__pycache__"))

        # Archive verification must work without Git and ignore hostile Git selectors.
        no_git_manifest = _run(
            archive,
            "release_manifest.py",
            "verify",
            environment={"PATH": ""},
        )
        no_git = _run(archive, "public_release_audit.py", environment={"PATH": ""})
        no_git_links = _run(
            archive,
            "check_local_links.py",
            environment={"PATH": ""},
        )
        hostile_git = _run(
            archive,
            "public_release_audit.py",
            environment={
                "GIT_DIR": str(REPOSITORY_ROOT / ".git"),
                "GIT_WORK_TREE": str(archive),
                "GIT_INDEX_FILE": str(REPOSITORY_ROOT / ".git" / "index"),
            },
        )
        assert no_git_manifest.returncode == 0, (
            no_git_manifest.stdout + no_git_manifest.stderr
        )
        assert no_git.returncode == 0, no_git.stdout + no_git.stderr
        assert no_git.stdout.strip().startswith("PUBLIC_RELEASE_AUDIT_PASS files=7 ")
        assert no_git_links.returncode == 0, no_git_links.stdout + no_git_links.stderr
        assert hostile_git.returncode == 0, hostile_git.stdout + hostile_git.stderr
        assert hostile_git.stdout.strip().startswith("PUBLIC_RELEASE_AUDIT_PASS files=7 ")

        (archive / "README.md").write_text("[status](STATUS.mx)\n", encoding="utf-8")
        for script, arguments in (
            ("release_manifest.py", ("verify",)),
            ("public_release_audit.py", ()),
            ("check_local_links.py", ()),
        ):
            completed = _run(archive, script, *arguments)
            assert completed.returncode == 1
            assert "SHA-256 mismatch: README.md" in completed.stdout + completed.stderr

        (archive / "README.md").write_text("[status](STATUS.md)\n", encoding="utf-8")
        (archive / "unexpected.txt").write_text("not declared\n", encoding="utf-8")
        for script, arguments in (
            ("release_manifest.py", ("verify",)),
            ("public_release_audit.py", ()),
            ("check_local_links.py", ()),
        ):
            completed = _run(archive, script, *arguments)
            assert completed.returncode == 1
            assert "unexpected=['unexpected.txt']" in completed.stdout + completed.stderr


def test_manifest_build_refuses_an_archive_authority() -> None:
    with tempfile.TemporaryDirectory(prefix=".phasepair-archive-", dir=REPOSITORY_ROOT) as raw:
        archive = Path(raw)
        scripts = archive / "scripts"
        scripts.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)
        _write_manifest(archive)
        completed = _run(archive, "release_manifest.py", "build")
        assert completed.returncode == 1
        assert "manifest build requires the exact Git checkout" in completed.stderr


def test_public_audit_accepts_valid_figure_exports_and_scans_png_text() -> None:
    with tempfile.TemporaryDirectory(prefix=".phaseset-figures-", dir=REPOSITORY_ROOT) as raw:
        archive = Path(raw)
        scripts = archive / "scripts"
        figures = archive / "figures"
        scripts.mkdir()
        figures.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)
        (archive / "README.md").write_text("figure fixture\n", encoding="utf-8")
        png_path = figures / "figure.png"
        png_path.write_bytes(_png_with_text(b"public PhaseSet figure"))
        (figures / "figure.pdf").write_bytes(b"%PDF-1.4\n%\x00binary\n%%EOF\n")
        _write_manifest(archive)

        clean = _run(archive, "public_release_audit.py")
        assert clean.returncode == 0, clean.stdout + clean.stderr

        png_path.write_bytes(_png_with_text(b"192.0.2.1" + b":" + b"33123"))
        _write_manifest(archive)
        rejected = _run(archive, "public_release_audit.py")
        assert rejected.returncode == 1
        assert "credential or endpoint pattern" in rejected.stdout + rejected.stderr


def test_public_audit_decodes_pdf_streams_and_rejects_active_content() -> None:
    with tempfile.TemporaryDirectory(prefix=".phaseset-pdf-", dir=REPOSITORY_ROOT) as raw:
        archive = Path(raw)
        scripts = archive / "scripts"
        figures = archive / "figures"
        scripts.mkdir()
        figures.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)
        (archive / "README.md").write_text("PDF audit fixture\n", encoding="utf-8")
        pdf = figures / "figure.pdf"

        hidden = b"C:" + b"\\" + b"restricted" + b"\\lineage.json"
        compressed = zlib.compress(hidden)
        pdf.write_bytes(
            b"%PDF-1.7\n1 0 obj\n<< /Length "
            + str(len(compressed)).encode()
            + b" /Filter /FlateDecode >>\nstream\n"
            + compressed
            + b"\nendstream\nendobj\n%%EOF\n"
        )
        _write_manifest(archive)
        rejected_stream = _run(archive, "public_release_audit.py")
        assert rejected_stream.returncode == 1
        assert "private/local locator" in rejected_stream.stdout + rejected_stream.stderr

        pdf.write_bytes(
            b"%PDF-1.7\n1 0 obj\n<< /Type /Filespec /EF << /F 2 0 R >> >>\n"
            b"endobj\n%%EOF\n"
        )
        _write_manifest(archive)
        rejected_attachment = _run(archive, "public_release_audit.py")
        assert rejected_attachment.returncode == 1
        assert "attachment or active content" in (
            rejected_attachment.stdout + rejected_attachment.stderr
        )


def test_public_audit_decodes_pdf_name_escapes_and_rejects_malformed_names() -> None:
    with tempfile.TemporaryDirectory(prefix=".phaseset-pdf-name-", dir=REPOSITORY_ROOT) as raw:
        archive = Path(raw)
        scripts = archive / "scripts"
        scripts.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)
        (archive / "escaped.pdf").write_bytes(
            b"%PDF-1.7\n1 0 obj\n<< /Open#41ction 2 0 R /Java#53cript (x) >>\n"
            b"endobj\n%%EOF\n"
        )
        _write_manifest(archive)
        escaped = _run(archive, "public_release_audit.py")
        assert escaped.returncode == 1
        assert "attachment or active content" in escaped.stdout + escaped.stderr

        (archive / "escaped.pdf").write_bytes(
            b"%PDF-1.7\n1 0 obj\n<< /Java#G0Script (x) >>\nendobj\n%%EOF\n"
        )
        _write_manifest(archive)
        malformed = _run(archive, "public_release_audit.py")
        assert malformed.returncode == 1
        assert "malformed PDF name escape" in malformed.stdout + malformed.stderr


def test_public_audit_rejects_png_ancillary_metadata_and_pptx_binary_media() -> None:
    with tempfile.TemporaryDirectory(prefix=".phaseset-container-", dir=REPOSITORY_ROOT) as raw:
        archive = Path(raw)
        scripts = archive / "scripts"
        scripts.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)

        png = archive / "metadata.png"
        png.write_bytes(_png_with_text(b"public").replace(
            struct.pack(">I", 0) + b"IEND",
            struct.pack(">I", 8)
            + b"eXIf"
            + b"metadata"
            + struct.pack(">I", zlib.crc32(b"eXIfmetadata") & 0xFFFFFFFF)
            + struct.pack(">I", 0)
            + b"IEND",
        ))
        _write_manifest(archive)
        png_rejected = _run(archive, "public_release_audit.py")
        assert png_rejected.returncode == 1
        assert "unsupported PNG ancillary" in png_rejected.stdout + png_rejected.stderr

        png.unlink()
        pptx = archive / "binary-media.pptx"
        with zipfile.ZipFile(pptx, "w") as package:
            package.writestr(
                "[Content_Types].xml",
                b'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
            )
            package.writestr(
                "_rels/.rels",
                b'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>',
            )
            package.writestr(
                "ppt/presentation.xml",
                b'<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>',
            )
            package.writestr("ppt/media/recording.mp4", b"opaque media bytes")
        _write_manifest(archive)
        pptx_rejected = _run(archive, "public_release_audit.py")
        assert pptx_rejected.returncode == 1
        assert "unsupported PPTX media" in pptx_rejected.stdout + pptx_rejected.stderr


def test_public_audit_recursively_decodes_svg_and_drawio_payloads() -> None:
    with tempfile.TemporaryDirectory(prefix=".phaseset-xml-", dir=REPOSITORY_ROOT) as raw:
        archive = Path(raw)
        scripts = archive / "scripts"
        scripts.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)
        hidden = b"C:" + b"\\" + b"restricted" + b"\\capture.bin"

        svg = archive / "encoded.svg"
        encoded = base64.b64encode(hidden).decode()
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><image href="data:text/plain;base64,'
            + encoded
            + '"/></svg>\n',
            encoding="utf-8",
        )
        _write_manifest(archive)
        svg_rejected = _run(archive, "public_release_audit.py")
        assert svg_rejected.returncode == 1
        assert "private/local locator" in svg_rejected.stdout + svg_rejected.stderr

        svg.unlink()
        nested = b'<mxGraphModel><root><mxCell value="' + hidden + b'"/></root></mxGraphModel>'
        quoted = urllib.parse.quote_from_bytes(nested).encode()
        compressor = zlib.compressobj(level=9, wbits=-15)
        compressed = compressor.compress(quoted) + compressor.flush()
        (archive / "encoded.drawio").write_text(
            '<mxfile><diagram>'
            + base64.b64encode(compressed).decode()
            + '</diagram></mxfile>\n',
            encoding="utf-8",
        )
        _write_manifest(archive)
        drawio_rejected = _run(archive, "public_release_audit.py")
        assert drawio_rejected.returncode == 1
        assert "private/local locator" in drawio_rejected.stdout + drawio_rejected.stderr


def test_git_mode_reads_index_bytes_and_paths_not_unstaged_worktree() -> None:
    with tempfile.TemporaryDirectory(prefix=".phasepair-index-", dir=REPOSITORY_ROOT) as raw:
        checkout = Path(raw)
        scripts = checkout / "scripts"
        scripts.mkdir()
        for name in SCRIPT_NAMES:
            shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts / name)
        (checkout / "README.md").write_text(
            "192.0.2.1" + ":" + "1234\n[missing](NOT_TRACKED.md)\n",
            encoding="utf-8",
        )
        (checkout / "RELEASE_FILES.sha256").write_text(
            MANIFEST_HEADER,
            encoding="utf-8",
            newline="\n",
        )
        assert _git(checkout, "init", "--quiet").returncode == 0
        assert _git(checkout, "add", "--all").returncode == 0
        built = _run(checkout, "release_manifest.py", "build")
        assert built.returncode == 0, built.stdout + built.stderr
        assert _git(checkout, "add", "RELEASE_FILES.sha256").returncode == 0

        # The clean-looking worktree must not mask the staged endpoint or broken link.
        (checkout / "README.md").write_text("clean worktree only\n", encoding="utf-8")
        assert _run(checkout, "release_manifest.py", "verify").returncode == 0
        (checkout / "README.md").unlink()
        assert _run(checkout, "release_manifest.py", "verify").returncode == 0
        audit = _run(checkout, "public_release_audit.py")
        links = _run(checkout, "check_local_links.py")
        assert audit.returncode == 1
        assert "credential or endpoint pattern in README.md" in audit.stdout
        assert links.returncode == 1
        assert "README.md:2: missing 'NOT_TRACKED.md'" in links.stdout
