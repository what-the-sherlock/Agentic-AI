import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile
from app.core.router import InputRouter
from app.api.schemas import AgentInput

@pytest.mark.asyncio
async def test_router_text_only():
    router = InputRouter()
    inp = AgentInput(text="hello world")
    routed = await router.route(inp.text, None)
    
    assert routed.extracted_text == "hello world"
    assert routed.meta["source"] == "text_only"

@pytest.mark.asyncio
async def test_router_handles_pdf_mocked():
    with patch("app.core.router.PDFService") as MockPDFService:
        mock_instance = MockPDFService.return_value
        mock_instance.extract_text_with_confidence.return_value = ("Fake PDF Content", 0.99)
        
        router = InputRouter()
        
        mock_file = MagicMock(spec=UploadFile)
        mock_file.content_type = "application/pdf"
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=b"%PDF-1.4...")
        
        routed = await router.route("some text", mock_file)
        
        assert routed.extracted_text == "Fake PDF Content"
        assert routed.meta["source"] == "pdf"