import urllib.request
import urllib.error
import json
import uuid

BASE_URL = "http://localhost:8000"

def make_request(path, data=None, files=None):
    url = f"{BASE_URL}{path}"
    headers = {'Content-Type': 'application/json'}
    
    if data:
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    elif files:
        # Multipart form data is harder with urllib, but let's try a simple version
        # for testing unsupported file type.
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
        
        body = []
        for key, (filename, content, content_type) in files.items():
            body.append(f'--{boundary}\r\n')
            body.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n')
            body.append(f'Content-Type: {content_type}\r\n\r\n')
            body.append(content.decode('utf-8') if isinstance(content, bytes) and 32 <= content[0] <= 126 else str(content))
            body.append('\r\n')
        
        # Add additional form fields (session_id)
        # Note: This is an oversimplification but should work for the status code test
        
        body.append(f'--{boundary}--\r\n')
        payload = "".join(body).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    else:
        req = urllib.request.Request(url, method='GET')
        
    try:
        with urllib.request.urlopen(req) as response:
            return response.getcode(), json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode('utf-8'))
        except:
            return e.code, {"detail": str(e)}
    except Exception as e:
        return 0, {"detail": str(e)}

def test_empty_question():
    print("\n[TEST] Empty Question")
    payload = {
        "question": "   ",
        "session_id": str(uuid.uuid4()),
        "chat_history": []
    }
    code, res = make_request("/chat", data=payload)
    print(f"Status: {code}")
    print(f"Detail: {res.get('detail')}")
    assert code == 400

def test_long_question():
    print("\n[TEST] Long Question (> 500 chars)")
    payload = {
        "question": "a" * 600,
        "session_id": str(uuid.uuid4()),
        "chat_history": []
    }
    code, res = make_request("/chat", data=payload)
    print(f"Status: {code}")
    # Detail may be highly nested for Pydantic errors
    print(f"Response: {res}")
    assert code == 422 

def test_invalid_session_id():
    print("\n[TEST] Invalid Session ID Format")
    payload = {
        "question": "Hello?",
        "session_id": "not-a-uuid",
        "chat_history": []
    }
    code, res = make_request("/chat", data=payload)
    print(f"Status: {code}")
    print(f"Detail: {res.get('detail')}")
    assert code == 400

if __name__ == "__main__":
    try:
        test_empty_question()
        test_long_question()
        test_invalid_session_id()
        print("\n✅ All basic chat logic edge case tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
