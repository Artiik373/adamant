"""The Hy4-flash dialect: the model's Jinja chat template with `reasoning_effort` no_think, high or max, and its
`<think:opensource>` tags read as the plain ones. Text only: its tool calls are not parsed."""
from __future__ import annotations

from serve import frontend
from serve.models import Dialect

THINK = {"<think:opensource>": "<think>", "</think:opensource>": "</think>"}
END = "<｜hy_end:opensource｜>"


class _Template:
    def __init__(self, source: str):
        self.inner = frontend.ChatTemplate(source)
        self.source = source

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        if kwargs.pop("enable_thinking", True) is False:
            kwargs["reasoning_effort"] = "no_think"
        return self.inner.render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


class _Parser(frontend.OutputParser):
    def feed(self, delta, *, final=False):
        for tag, plain in THINK.items():
            delta = delta.replace(tag, plain)
        return super().feed(delta, final=final)

    def _quoted_think_end(self, i):
        return False        # the tag is one token, never quoted text


class Hy4(Dialect):
    name = "hy4"
    stop_texts = (END,)
    reasoning_close = "</think:opensource>"
    reasoning_wrap_up = "\n\nI have thought about this long enough; time to give my answer.</think:opensource>"

    def effort_level(self, level):
        if isinstance(level, int):
            raise ValueError(f"unknown reasoning effort {level!r}: use none, low, medium, high or max")
        return "max" if level == "max" else "high"

    def template(self, source):
        return _Template(source)

    def parser(self, thinking, tools, stream_tools=True):
        return _Parser(thinking=thinking, tools=None, stream_tools=stream_tools)

    def literal_tags(self, controls):
        tags = frontend.literal_tags(controls)
        top = max(map(ord, tags.values()))      # after the control marks
        tags.update({tag: chr(top + 1 + k) for k, tag in enumerate(THINK)})
        return dict(sorted(tags.items(), key=lambda t: -len(t[0])))

    def forced_opening(self, name):
        raise ValueError("this model's tool calls are not supported: drop tool_choice")
