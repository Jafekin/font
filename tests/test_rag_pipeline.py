from rag.pipeline import RAGPipeline, set_analyze_backend
from unittest import TestCase
from pathlib import Path
import sys
import os
os.environ.setdefault("RAG_FAKE_EMBEDDINGS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class RagPipelineTests(TestCase):
    def setUp(self):
        # Prepare a real image path from tests assets
        self.image_path = str(PROJECT_ROOT / "tests" / "image" / "22.jpg")
        assert os.path.isfile(self.image_path), "Test image missing"
        # Inject fake analysis backend to avoid real LLM calls

        def _fake_backend(image, script_type, hint="", prompt=None):
            return {
                "document_metadata": {"document_type": {"value": script_type, "confidence": 0.8}},
                "used_references": ["fake-0"],
                "prompt": prompt or "",
            }

        set_analyze_backend(_fake_backend)

    def tearDown(self):
        set_analyze_backend(None)

    def test_run_success(self):
        pipeline = RAGPipeline(index_path=str(PROJECT_ROOT / "results"))
        out = pipeline.run(self.image_path, script_type="汉文古籍", hint="史记")
        self.assertTrue(out.get("success"))
        self.assertIn("analysis", out)
        self.assertEqual(out.get("num_references"), 5)
        self.assertEqual(len(out.get("retrieved_references", [])), 5)
        self.assertEqual(len(out.get("retrieval_scores", [])), 5)

    def test_search_similar_by_image(self):
        pipeline = RAGPipeline(index_path=str(PROJECT_ROOT / "results"))
        res = pipeline.search_similar(query_image_path=self.image_path, k=3)
        self.assertEqual(len(res), 3)
        self.assertTrue(all("id" in r and "score" in r for r in res))

    def test_search_similar_by_text(self):
        pipeline = RAGPipeline(index_path=str(PROJECT_ROOT / "results"))
        res = pipeline.search_similar(query_text="司马迁 史记", k=2)
        self.assertEqual(len(res), 2)

    def test_batch_analyze(self):
        pipeline = RAGPipeline(index_path=str(PROJECT_ROOT / "results"))
        images = [self.image_path, self.image_path]
        outs = pipeline.batch_analyze(
            images, script_type="汉文古籍", hints=["a", "b"], k=4)
        self.assertEqual(len(outs), 2)
        self.assertTrue(all(o.get("success") for o in outs))
        self.assertTrue(all(o.get("num_references") == 4 for o in outs))
