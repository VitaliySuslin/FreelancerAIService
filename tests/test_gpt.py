import pytest
from unittest.mock import AsyncMock, patch

from ..app.src.integrations.gpt import GPTClient
from ..app.src.app import FreelancerAnalyzer


@pytest.mark.asyncio
@patch("g4f.client.Client.chat.completions.create")
async def test_generate_response_success(mock_create):
    mock_create.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    
    client = GPTClient()
    response = await client.generate_response("Test prompt")
    
    assert response == "Test response"
    mock_create.assert_called_once()

@pytest.mark.asyncio
@patch("g4f.client.Client.chat.completions.create")
async def test_generate_response_failure(mock_create):
    mock_create.side_effect = Exception("API error")
    
    client = GPTClient()
    response = await client.generate_response("Test prompt")
    
    assert response is None 

@pytest.mark.asyncio
def test_load_and_query_documents():
    client = GPTClient()
    client.load_documents("tests/test_data.csv")
    
    response = client.query_documents("What is the total?")
    assert "total" in response.lower()

@patch("kagglehub.dataset_download")
@patch("app.src.integrations.gpt.GPTClient")
def test_analyzer(mock_gpt, mock_download):
    analyzer = FreelancerAnalyzer()
    analyzer.load_data()
    analyzer.ask_question("Test question")
    
    mock_download.assert_called_once()
    mock_gpt.return_value.load_documents.assert_called_once()
    mock_gpt.return_value.query_documents.assert_called_once_with("Test question")