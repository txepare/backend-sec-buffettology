import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.sec_api import SecEdgarAPI
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

# 1. Test root
r_root = client.get('/')
assert r_root.status_code == 200, f'Root error: {r_root.text}'
print('Root endpoint OK:', r_root.json())

# 2. Test search endpoint
r_search = client.get('/buscar?q=coca')
assert r_search.status_code == 200
print('Search endpoint OK (/buscar?q=coca):', r_search.json())

# 3. Test resolution
for test_input in ['Apple', 'AAPL', 'coca-cola', 'KO', 'Google', 'Facebook', 'Berkshire']:
    comp = SecEdgarAPI.resolve_company(test_input)
    print(f"Resolution: '{test_input}' -> {comp['ticker']} ({comp['title']})")
