import os
os.environ.setdefault("RAG_FAKE_EMBEDDINGS", "1")
import sys
from pathlib import Path
from unittest import TestCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class EmbeddingsTests(TestCase):
    def setUp(self):
        self.image_path = str(PROJECT_ROOT / "tests" / "image" / "22.jpg")
        assert os.path.isfile(self.image_path), "Test image missing"

    def test_image_and_text_embeddings(self):
        from rag.embeddings import get_image_embedding, get_text_embedding
        vec_img = get_image_embedding(self.image_path)
        vec_txt = get_text_embedding("测试文本")
        self.assertEqual(len(vec_img), 512)
        self.assertEqual(len(vec_txt), 512)
        self.assertTrue(all(isinstance(x, float) for x in vec_img))
        self.assertTrue(all(isinstance(x, float) for x in vec_txt))

    def test_batch_embeddings(self):
        from rag.embeddings import batch_image_embeddings, batch_text_embeddings
        imgs = [self.image_path, self.image_path]
        texts = ["甲骨文", "敦煌"]
        vimg = batch_image_embeddings(imgs)
        vtxt = batch_text_embeddings(texts)
        self.assertEqual(len(vimg), 2)
        self.assertEqual(len(vtxt), 2)
        self.assertEqual(len(vimg[0]), 512)
        self.assertEqual(len(vtxt[0]), 512)
