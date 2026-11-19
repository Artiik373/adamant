"""Hy4-flash packs: the tokenizer files derived from tokenizer.json, and the dialect's template and think tags."""
import json
import sys
import unittest
from pathlib import Path

from serve.engine import modelfile
from serve.models import dialect_for

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import tokenizer as ST  # noqa: E402

HF = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "Ġ": 3}, "merges": [["a", "b"]]},
      "added_tokens": [{"id": 4, "content": "<end>", "special": True},
                       {"id": 5, "content": "<think:opensource>", "special": False}]}
TEMPLATE = ("{% for m in messages %}{{ m.content }}{% endfor %}"
            "{% if add_generation_prompt %}<{{ reasoning_effort | default('high') }}>{% endif %}")


class Hy4Test(unittest.TestCase):
    def test_tokenizer_files(self):
        f = modelfile.hf_tokenizer_files(json.dumps(HF).encode(), "hy4")
        vocab = json.loads(f["vocab.json"])
        self.assertEqual(vocab["<end>"], 4)
        self.assertEqual(json.loads(f["token_type.json"]), [1, 1, 1, 1, 3, 4])
        self.assertEqual(f["merges.txt"], b"a b")
        self.assertEqual(json.loads(f["tokenizer.json"]), {"pre": "hy4"})
        toks = [None] * len(vocab)
        for t, i in vocab.items():
            toks[i] = t
        tk = ST.Tokenizer(toks, ["a b"], json.loads(f["token_type.json"]), pre="hy4")
        self.assertEqual(tk.encode("ab<think:opensource>a<end>", parse_special=True), [2, 5, 0, 4])

    def test_dialect(self):
        d = dialect_for("hy4")
        t = d.template(TEMPLATE)
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(t.render(msgs), "hi<high>")
        self.assertEqual(t.render(msgs, enable_thinking=False), "hi<no_think>")
        self.assertEqual(d.effort_level("low"), "high")
        p = d.parser(True, None)
        evs = p.feed('say "Paris."') + p.feed("</think:opensource>") + p.feed("Paris.") + p.finish()
        self.assertEqual([(e.kind, e.text) for e in evs], [("reasoning", 'say "Paris."'), ("content", "Paris.")])
        marks = d.literal_tags(["<end>"])
        self.assertEqual(len(set(marks.values())), len(marks))


if __name__ == "__main__":
    unittest.main()
