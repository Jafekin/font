from unittest import TestCase
from rag.prompt import get_prompt


class PromptTests(TestCase):
    def test_get_prompt_includes_values(self):
        prompt = get_prompt("汉文古籍", "提示词", ["上下文1", "上下文2"])
        self.assertIn("汉文古籍", prompt)
        self.assertIn("提示词", prompt)
        self.assertIn("上下文1", prompt)
        self.assertIn("上下文2", prompt)

    def test_get_prompt_when_no_context(self):
        prompt = get_prompt("甲骨文", "", [])
        self.assertIn("甲骨文", prompt)
        # Ensure placeholder [] used
        self.assertIn("[]", prompt)
