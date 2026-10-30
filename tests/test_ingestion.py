from app.ingestion.pdf_loader import PDFLoader


def test_pdf_loader_extracts_pages_and_coordinates(sample_pdf_path):
    extracted = PDFLoader.load_pdf(sample_pdf_path)

    assert extracted.filename == "sample_benchmark_doc.pdf"
    assert extracted.num_pages == 2
    assert len(extracted.blocks) >= 4
    assert extracted.total_characters > 100

    # Verify coordinates on blocks
    for b in extracted.blocks:
        assert b.page_number in [1, 2]
        assert len(b.bbox) == 4
        x0, y0, x1, y1 = b.bbox
        assert x1 > x0
        assert y1 > y0
        assert len(b.text.strip()) > 0
        assert len(b.spans) > 0

    # Check page dimensions
    assert 1 in extracted.page_dimensions
    assert 2 in extracted.page_dimensions
    w1, h1 = extracted.page_dimensions[1]
    assert w1 == 600.0 and h1 == 800.0
