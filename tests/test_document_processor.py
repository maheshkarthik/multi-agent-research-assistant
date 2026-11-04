"""Tests for src.local_docs.document_processor (copied from Project 01).

Run from the project root with:  python -m unittest -v
"""

import io
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from src.local_docs.document_processor import (
    DocumentProcessingError,
    DocumentProcessor,
    split_text,
)


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal PDF where each page shows the given lines of text.

    An empty list of lines produces a page with no text.
    """

    def escape(line: str) -> str:
        return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def stream(data: bytes) -> bytes:
        return b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"

    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % pid for pid in page_ids), len(pages)),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, lines in zip(page_ids, pages):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (page_id + 1)
        )
        shows = " ".join(f"({escape(line)}) Tj T*" for line in lines)
        content = f"BT /F1 10 Tf 12 TL 50 750 Td {shows} ET" if lines else ""
        objects.append(stream(content.encode("latin-1")))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_offset = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return bytes(out)


def long_page(page: int, line_count: int = 50) -> list[str]:
    """About 3,000 characters of distinct lines, enough for several chunks."""
    return [f"Page {page} line {i:03d}: the quick brown fox jumps over it." for i in range(line_count)]


class SplitTextTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("hello world", 1000, 200), ["hello world"])

    def test_blank_text_gives_no_chunks(self):
        self.assertEqual(split_text("   \n\t ", 1000, 200), [])

    def test_chunks_respect_size(self):
        chunks = split_text("x" * 3500, 1000, 200)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))
        self.assertEqual(len(chunks[0]), 1000)

    def test_consecutive_chunks_overlap_by_exactly_overlap_chars(self):
        text = "".join(chr(ord("a") + i % 26) for i in range(3500))
        chunks = split_text(text, 1000, 200)
        self.assertGreater(len(chunks), 1)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous[-200:], current[:200])

    def test_chunks_reassemble_to_original_text(self):
        text = "".join(str(i % 10) for i in range(3500))
        chunks = split_text(text, 1000, 200)
        rebuilt = chunks[0] + "".join(c[200:] for c in chunks[1:])
        self.assertEqual(rebuilt, text)

    def test_no_redundant_trailing_chunk(self):
        # 1,800 chars: chunk at 0 (0-1000) and 800 (800-1800) covers everything.
        self.assertEqual(len(split_text("y" * 1800, 1000, 200)), 2)

    def test_invalid_settings_raise(self):
        for size, overlap in [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    split_text("text", size, overlap)
                with self.assertRaises(ValueError):
                    DocumentProcessor(size, overlap)


class ProcessPdfTests(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

    def test_extracts_text_from_pdf(self):
        pdf = make_pdf([["Quarterly revenue grew 12% (year over year)."]])
        document = self.processor.process_pdf(pdf, source_name="report.pdf")
        self.assertEqual(document.page_count, 1)
        self.assertEqual(len(document.chunks), 1)
        self.assertIn("Quarterly revenue grew 12% (year over year).", document.chunks[0].text)

    def test_metadata_on_every_chunk(self):
        pdf = make_pdf([long_page(1), ["Page 2 is short."], long_page(3)])
        document = self.processor.process_pdf(pdf, source_name="handbook.pdf")

        self.assertEqual(document.page_count, 3)
        self.assertEqual({c.page for c in document.chunks}, {1, 2, 3})
        self.assertEqual([c.chunk_index for c in document.chunks], list(range(len(document.chunks))))
        for chunk in document.chunks:
            self.assertEqual(chunk.source, "handbook.pdf")
            self.assertIn(f"Page {chunk.page} ", chunk.text)
            self.assertEqual(
                chunk.metadata,
                {"source": "handbook.pdf", "page": chunk.page, "chunk_index": chunk.chunk_index},
            )

    def test_chunks_overlap_within_a_page(self):
        document = self.processor.process_pdf(make_pdf([long_page(1)]), source_name="a.pdf")
        chunks = document.chunks
        self.assertGreater(len(chunks), 2)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.text[-200:], current.text[:200])

    def test_chunks_do_not_cross_pages(self):
        document = self.processor.process_pdf(make_pdf([long_page(1), long_page(2)]), source_name="a.pdf")
        for chunk in document.chunks:
            other_page = 2 if chunk.page == 1 else 1
            self.assertNotIn(f"Page {other_page} ", chunk.text)

    def test_respects_custom_chunk_settings(self):
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
        document = processor.process_pdf(make_pdf([long_page(1)]), source_name="a.pdf")
        self.assertTrue(all(len(c.text) <= 300 for c in document.chunks))
        for previous, current in zip(document.chunks, document.chunks[1:]):
            self.assertEqual(previous.text[-50:], current.text[:50])

    def test_pages_without_text_are_skipped(self):
        pdf = make_pdf([["First page text."], [], ["Third page text."]])
        document = self.processor.process_pdf(pdf, source_name="mixed.pdf")
        self.assertEqual(document.page_count, 3)
        self.assertEqual(document.skipped_pages, [2])
        self.assertEqual([c.page for c in document.chunks], [1, 3])
        self.assertEqual([c.chunk_index for c in document.chunks], [0, 1])

    def test_pdf_with_only_blank_pages_is_empty(self):
        document = self.processor.process_pdf(make_pdf([[], []]), source_name="blank.pdf")
        self.assertTrue(document.is_empty)
        self.assertEqual(document.skipped_pages, [1, 2])

    def test_pdf_with_zero_pages_is_empty(self):
        buffer = io.BytesIO()
        PdfWriter().write(buffer)
        document = self.processor.process_pdf(buffer.getvalue(), source_name="nothing.pdf")
        self.assertEqual(document.page_count, 0)
        self.assertTrue(document.is_empty)

    def test_invalid_pdf_raises(self):
        for name, data in [("text.pdf", b"this is not a pdf"), ("empty.pdf", b""), ("cut.pdf", make_pdf([["x"]])[:40])]:
            with self.subTest(name=name):
                with self.assertRaises(DocumentProcessingError):
                    self.processor.process_pdf(data, source_name=name)

    def test_missing_file_raises(self):
        with self.assertRaises(DocumentProcessingError):
            self.processor.process_pdf("does/not/exist.pdf")

    def test_encrypted_pdf_raises(self):
        writer = PdfWriter()
        writer.add_blank_page(612, 792)
        writer.encrypt(user_password="secret", algorithm="RC4-128")
        buffer = io.BytesIO()
        writer.write(buffer)
        with self.assertRaises(DocumentProcessingError):
            self.processor.process_pdf(buffer.getvalue(), source_name="locked.pdf")

    def test_source_name_inferred_from_path_and_file_object(self):
        pdf = make_pdf([["Some text."]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "from_disk.pdf"
            path.write_bytes(pdf)
            self.assertEqual(self.processor.process_pdf(path).chunks[0].source, "from_disk.pdf")
            self.assertEqual(self.processor.process_pdf(str(path)).source, "from_disk.pdf")

        upload = io.BytesIO(pdf)
        upload.name = "uploaded.pdf"  # Streamlit's UploadedFile exposes .name the same way
        self.assertEqual(self.processor.process_pdf(upload).source, "uploaded.pdf")

    def test_chroma_records_are_parallel_and_unique(self):
        pdf = make_pdf([long_page(1), long_page(2)])
        records = self.processor.process_pdf(pdf, source_name="a.pdf").to_chroma_records()
        self.assertEqual(set(records), {"ids", "documents", "metadatas"})
        self.assertEqual(len(records["ids"]), len(records["documents"]))
        self.assertEqual(len(records["ids"]), len(records["metadatas"]))
        self.assertEqual(len(set(records["ids"])), len(records["ids"]))
        for metadata in records["metadatas"]:
            self.assertTrue(all(isinstance(v, (str, int, float, bool)) for v in metadata.values()))


if __name__ == "__main__":
    unittest.main()
