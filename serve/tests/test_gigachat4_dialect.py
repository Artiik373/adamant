"""The GigaChat 4 dialect: template kwargs, stop tokens, and incremental GCML tool-call parsing."""
import unittest
from pathlib import Path

from serve.models import dialect_for
from serve.models.gigachat4 import (BOS, CALLS_END, CALLS_START, EOS, GCML_TOKEN, INVOKE_START, MESSAGE_SEP,
                                    PARAM_END, PARAM_START, GigaChat4)

PACK = Path("G:/llm/models/gigachat4-nvfp4.basalt")


class GigaChat4DialectTest(unittest.TestCase):
    def test_registry_and_kwargs(self):
        d = dialect_for("gigachat4")
        self.assertIsInstance(d, GigaChat4)
        self.assertEqual(d.stop_texts, (MESSAGE_SEP, EOS))
        self.assertEqual(d.template_kwargs({}), {"bos_token": BOS})
        self.assertEqual(d.template_kwargs({"enable_thinking": False, "preserve_reasoning": False}),
                         {"bos_token": BOS, "reasoning": False, "drop_history_thinking": True})

    def test_parser_reasoning_content_and_gcml_calls(self):
        p = GigaChat4().parser(thinking=True, tools=[{"name": "calc"}])
        text = (
            "let me check</think>\n\nResult:\n\n"
            f"{CALLS_START}\n"
            f'{INVOKE_START} name="calc">\n'
            f'{PARAM_START} name="expr" string="true">1+2{PARAM_END}\n'
            f'{PARAM_START} name="prec" string="false">4{PARAM_END}\n'
            f"</{GCML_TOKEN}invoke>\n"
            f"{CALLS_END}"
        )
        evs = []
        for ch in text:
            evs.extend(p.feed(ch))
        evs.extend(p.finish())
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "let me check")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "\n\nResult:")
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "calc")
        self.assertEqual(calls[0].arguments, {"expr": "1+2", "prec": 4})

    @unittest.skipUnless(PACK.exists(), "gigachat4 pack not present")
    def test_pack_chat_template_renders(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from model_file import ModelFile
        src = ModelFile(str(PACK)).read("tokenizer/chat_template.jinja").decode("utf-8")
        d = GigaChat4()
        tpl = d.template(src)
        p_on = tpl.render([{"role": "user", "content": "hi"}], **d.template_kwargs({}))
        p_off = tpl.render([{"role": "user", "content": "hi"}], **d.template_kwargs({"enable_thinking": False}))
        self.assertTrue(p_on.startswith("<s>system<|role_sep|>\n"))
        self.assertTrue(p_on.endswith("assistant<|role_sep|>\n<think>"))
        self.assertTrue(p_off.endswith("assistant<|role_sep|>\n<think></think>\n\n"))
