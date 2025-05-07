from fastapi.testclient import TestClient
from src.api.api import app

client = TestClient(app)

def test_ask_question():
    response = client.post(
        "/ask",
        json={"question": "什么是注意力机制？", "paper_id": "123"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "recommendations" in response.json()

def test_upload_paper():
    with open("test.pdf", "rb") as f:
        response = client.post(
            "/upload_paper",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    assert "paper_id" in response.json()
    


'''
# 问答API测试
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"神经网络原理","paper_id":"123"}'

# 文件上传测试
curl -X POST "http://localhost:8000/upload_paper" \
-H "Content-Type: multipart/form-data" \
-F "file=@paper.pdf"
'''


if __name__ == "__main__":
    '''
    python -m tests.test_api
    '''
    test_ask_question()
    # test_upload_paper()