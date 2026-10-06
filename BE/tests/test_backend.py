import io
import time
import unittest
from fastapi.testclient import TestClient

from main import app
from app.extractors.pdf import extract_pdf
from app.extractors.docx import extract_docx
from app.extractors.excel import extract_excel
from app.extractors.txt import extract_txt
from app.processing.chunker import chunk_document_blocks

client = TestClient(app)

class TestBackendPipeline(unittest.TestCase):

    def test_txt_extractor_and_chunker(self):
        sample_text = b"Heading 1\n\nThis is paragraph one.\n\nThis is paragraph two."
        extracted = extract_txt(sample_text)
        self.assertEqual(len(extracted), 1)
        self.assertIn("Heading 1", extracted[0]["text"])

        chunks = chunk_document_blocks(
            blocks=extracted,
            doc_id="test_doc_1",
            file_name="test.txt",
            strategy="recursive",
            chunk_size=100,
            chunk_overlap=20
        )
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].metadata["file_name"], "test.txt")
        self.assertEqual(chunks[0].metadata["doc_id"], "test_doc_1")

    def test_fastapi_upload_and_query_flow(self):
        # 1. Upload TXT file
        file_content = b"FastAPI with ChromaDB allows document ingestion and vector similarity search. Vector embeddings enable RAG."
        response = client.post(
            "/documents/upload",
            files={"file": ("rag_info.txt", file_content, "text/plain")},
            data={"strategy": "recursive", "chunk_size": 200, "chunk_overlap": 20}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("task_id", data)
        self.assertIn("doc_id", data)

        task_id = data["task_id"]
        doc_id = data["doc_id"]

        # 2. Wait briefly for background task completion
        time.sleep(1.5)
        status_resp = client.get(f"/documents/tasks/{task_id}")
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.json()["status"], "COMPLETED")

        # 3. Query documents
        query_resp = client.post(
            "/documents/query",
            json={"query": "What allows vector similarity search?", "top_k": 2}
        )
        self.assertEqual(query_resp.status_code, 200)
        query_data = query_resp.json()
        self.assertGreater(query_data["results_count"], 0)
        self.assertIn("FastAPI", query_data["results"][0]["content"])

        # 4. List documents
        list_resp = client.get("/documents")
        self.assertEqual(list_resp.status_code, 200)
        docs = list_resp.json()["documents"]
        doc_ids = [d["doc_id"] for d in docs]
        self.assertIn(doc_id, doc_ids)

        # 5. Delete document
        delete_resp = client.delete(f"/documents/{doc_id}")
        self.assertEqual(delete_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
