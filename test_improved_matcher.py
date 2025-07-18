#!/usr/bin/env python3
"""
Test script for the improved matcher with typo correction and prioritized objection matching.
"""

import requests
import json
import time

def test_improved_matcher():
    """Test the improved matcher with the specific failing case."""
    
    # Wait for server to start
    time.sleep(3)
    
    # Test the specific case that was failing
    test_case = {
        "prompt": "3sol",
        "text": "I don't want a solo"
    }
    
    print("Testing improved matcher...")
    print(f"Input: {test_case}")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/match_intent",
            json=test_case,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Response: {result}")
            
            # Check if it now correctly matches the objection
            if result.get('action') == 'wav_response_hangup' and 'solar' in result.get('matched_intent', '').lower():
                print("✓ SUCCESS: Now correctly matches solar objection!")
            else:
                print("⚠ Still not matching correctly")
                
            return result
        else:
            print(f"✗ API call failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return None

def test_additional_cases():
    """Test additional cases to verify the improvements."""
    
    test_cases = [
        {
            "name": "Typo correction - solo to solar",
            "prompt": "3sol",
            "text": "I don't want a solo",
            "expected_action": "wav_response_hangup"
        },
        {
            "name": "Multiple typos",
            "prompt": "2sol",
            "text": "how expencive is soler",
            "expected_action": "wav_response"
        },
        {
            "name": "Price question",
            "prompt": "2sol",
            "text": "how expensive is it?",
            "expected_action": "wav_response"
        },
        {
            "name": "TPS objection",
            "prompt": "2sol",
            "text": "I'm on the TPS list",
            "expected_action": "wav_response_hangup"
        }
    ]
    
    print("\n" + "="*50)
    print("TESTING ADDITIONAL CASES")
    print("="*50)
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Input: {test_case['text']}")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/match_intent",
                json={"prompt": test_case['prompt'], "text": test_case['text']},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Action: {result.get('action')}")
                print(f"Confidence: {result.get('confidence', 0):.3f}")
                print(f"Matched Intent: {result.get('matched_intent')}")
                
                if result.get('action') == test_case['expected_action']:
                    print("✓ Expected action matched!")
                else:
                    print(f"⚠ Expected {test_case['expected_action']}, got {result.get('action')}")
            else:
                print(f"✗ API call failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
        
        time.sleep(0.5)  # Small delay between tests

if __name__ == "__main__":
    print("Testing Improved Matcher with Typo Correction and Prioritized Objection Matching")
    print("="*80)
    
    # Test the main failing case
    result = test_improved_matcher()
    
    # Test additional cases
    test_additional_cases()
    
    print("\n" + "="*80)
    print("Test completed!") 