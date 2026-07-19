import pytest
from app import app  # Import your Flask app instance

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_recommend_endpoint(client):
    # Simulate a GET request to the /recommend route
    response = client.get('/recommend?userId=1')
    if response.status_code == 500:
        print(response.data)
    
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200
    
    # Assert that the response is in JSON format
    assert response.is_json
    
    # Optionally check if the response data is a list
    data = response.get_json()
    assert isinstance(data, list)