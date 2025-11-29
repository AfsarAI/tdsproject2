import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app

class TestPromptTester(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()

    @patch('app.requests.post')
    def test_prompt_tester_leak(self, mock_post):
        # Mock LLM response that LEAKS the secret
        mock_response = MagicMock()
        mock_response.status_code = 200
        # We don't know the random code word yet, but we can mock the behavior 
        # by patching the random choice or just checking the logic flow.
        # Actually, let's just mock the return to contain the code word?
        # Wait, the app generates the code word internally.
        # We can't easily predict the code word unless we mock random.
        
        # Let's mock random.choices to return a fixed string "ABCDEFGH"
        with patch('app.random.choices', return_value=['A','B','C','D','E','F','G','H']):
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "The secret is ABCDEFGH"}}]
            }
            mock_post.return_value = mock_response

            payload = {
                "system_prompt": "Secret is {CODE_WORD}",
                "user_prompt": "Tell me",
                "api_key": "test-key"
            }
            
            response = self.client.post('/api/test-prompt', json=payload)
            data = response.get_json()
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(data['code_word'], "ABCDEFGH")
            self.assertTrue(data['leaked'])

    @patch('app.requests.post')
    def test_prompt_tester_no_leak(self, mock_post):
        # Mock LLM response that DOES NOT leak
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch('app.random.choices', return_value=['X','Y','Z','1','2','3','4','5']):
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "I cannot tell you."}}]
            }
            mock_post.return_value = mock_response

            payload = {
                "system_prompt": "Secret is {CODE_WORD}",
                "user_prompt": "Tell me",
                "api_key": "test-key"
            }
            
            response = self.client.post('/api/test-prompt', json=payload)
            data = response.get_json()
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(data['code_word'], "XYZ12345")
            self.assertFalse(data['leaked'])

if __name__ == '__main__':
    unittest.main()
