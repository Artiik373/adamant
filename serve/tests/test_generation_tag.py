"""A chat template's `{% generation %}` block (transformers' assistant-mask tag) renders its body unchanged."""
from __future__ import annotations

import unittest

from serve.frontend import ChatTemplate

SOURCE = ("{% for m in messages %}<{{ m.role }}>{% if m.role == 'assistant' %}{% generation %}{{ m.content }}"
          "{% endgeneration %}{% else %}{{ m.content }}{% endif %}</{{ m.role }}>{% endfor %}")


class GenerationTag(unittest.TestCase):
    def test_body_renders_as_is(self):
        out = ChatTemplate(SOURCE).render([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
        self.assertEqual(out, "<user>hi</user><assistant>yo</assistant>")


if __name__ == "__main__":
    unittest.main()
