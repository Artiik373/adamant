"""The GigaChat 4 dialect: chatml v5 Jinja template, `<think>` reasoning and GCML tool calls."""
from __future__ import annotations

import json
import re

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect

BOS = "<s>"
EOS = "</s>"
ROLE_SEP = "<|role_sep|>\n"
MESSAGE_SEP = "<|message_sep|>\n\n"
THINK_END = "</think>"
GCML_TOKEN = "｜GCML｜"
GCML_MARK = "\U000F2600"
CALLS_START = f"<{GCML_TOKEN}tool_calls>"
CALLS_END = f"</{GCML_TOKEN}tool_calls>"
INVOKE_START = f"<{GCML_TOKEN}invoke"
INVOKE_END = f"</{GCML_TOKEN}invoke>"
PARAM_START = f"<{GCML_TOKEN}parameter"
PARAM_END = f"</{GCML_TOKEN}parameter>"
CALLS_GAP = "\n\n"
INVOKE_HEAD = re.compile(r"^" + re.escape(INVOKE_START) + r' name="([^"]*)">')
PARAM_HEAD = re.compile(r"^" + re.escape(PARAM_START) + r' name="([^"]*)" string="(true|false)">')


def _value(text: str, is_string: bool):
    if is_string:
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


class GigaChat4Parser:
    """Incremental parser of GigaChat 4 text into reasoning, content, and GCML tool calls."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.stream_tools = stream_tools
        self.sub = "between"            # inside a call block: between | invoke | value
        self.call: ToolCall | None = None
        self.args: dict = {}
        self.first = True
        self.pname, self.pstring, self.pvalue = "", True, ""
        self.calls_seen = False

    def _args(self, out: list, piece: str) -> None:
        if piece and self.stream_tools:
            out.append(Event("tool_args", piece, call=self.call))

    def _end_call(self, out: list) -> None:
        self._args(out, "}")
        out.append(Event("tool_call", call=ToolCall(name=self.call.name, arguments=self.args, id=self.call.id)))
        self.call = None

    def _reasoning(self, out: list) -> bool:
        end, calls = self.buf.find(THINK_END), self.buf.find(CALLS_START)
        if end >= 0 and (calls < 0 or end < calls):
            if end:
                out.append(Event("reasoning", self.buf[:end]))
            self.buf, self.state = self.buf[end + len(THINK_END):], "content"
            return True
        if calls >= 0:
            text = self.buf[:calls].rstrip()
            if text:
                out.append(Event("reasoning", text))
            self.buf, self.state = self.buf[calls + len(CALLS_START):], "call"
            return True
        keep = frontend.hold_back(self.buf, (THINK_END, CALLS_GAP + CALLS_START, CALLS_START))
        if len(self.buf) > keep:
            out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _content(self, out: list) -> bool:
        calls = self.buf.find(CALLS_START)
        if calls >= 0:
            text = self.buf[:calls].rstrip("\n")
            if text:
                out.append(Event("content", text))
            self.buf, self.state = self.buf[calls + len(CALLS_START):], "call"
            return True
        keep = frontend.hold_back(self.buf, (CALLS_GAP + CALLS_START, CALLS_START))
        if len(self.buf) > keep:
            out.append(Event("content", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _call(self, out: list) -> bool:
        if self.sub == "between":
            s = self.buf.lstrip()
            if s.startswith(CALLS_END):
                self.buf, self.state, self.calls_seen = s[len(CALLS_END):], "content", True
                return True
            m = INVOKE_HEAD.match(s)
            if m:
                self.call = ToolCall(name=m.group(1), arguments={})
                self.args, self.first, self.sub = {}, True, "invoke"
                self.buf = s[m.end():]
                if self.stream_tools:
                    out.append(Event("tool_start", call=self.call))
                self._args(out, "{")
                return True
            if s and not (CALLS_END.startswith(s) or INVOKE_START.startswith(s) or
                          (s.startswith(INVOKE_START) and ">" not in s)):
                self.buf, self.state = CALLS_START + self.buf, "content"
                self.sub, self.calls_seen = "between", True
                out.append(Event("content", self.buf))
                self.buf = ""
            return False
        if self.sub == "invoke":
            s = self.buf.lstrip()
            if s.startswith(INVOKE_END):
                self.buf, self.sub = s[len(INVOKE_END):], "between"
                self._end_call(out)
                return True
            m = PARAM_HEAD.match(s)
            if m:
                self.pname, self.pstring, self.pvalue = m.group(1), m.group(2) == "true", ""
                self._args(out, ("" if self.first else ",") + json.dumps(m.group(1), ensure_ascii=False) + ":")
                self.first = False
                if self.pstring:
                    self._args(out, '"')
                self.buf, self.sub = s[m.end():], "value"
                return True
            return False
        end = self.buf.find(PARAM_END)
        if end >= 0:
            text = self.buf[:end]
            self.pvalue += text
            self.args[self.pname] = _value(self.pvalue, self.pstring)
            if self.pstring:
                self._args(out, json.dumps(text, ensure_ascii=False)[1:-1] + '"')
            else:
                self._args(out, json.dumps(self.args[self.pname], ensure_ascii=False))
            self.buf, self.sub = self.buf[end + len(PARAM_END):], "invoke"
            return True
        keep = frontend.hold_back(self.buf, (PARAM_END,))
        text, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        self.pvalue += text
        if self.pstring:
            self._args(out, json.dumps(text, ensure_ascii=False)[1:-1])
        return False

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "reasoning":
                more = self._reasoning(out)
            elif self.state == "content":
                more = self._content(out)
            else:
                more = self._call(out)
            if not more:
                return out

    def finish(self) -> list[Event]:
        out: list[Event] = []
        if self.state == "call":
            if self.call is not None:
                if self.sub == "value":
                    self.pvalue += self.buf
                    if self.pstring:
                        self._args(out, json.dumps(self.buf, ensure_ascii=False)[1:-1] + '"')
                    else:
                        self._args(out, json.dumps(self.pvalue))
                self._args(out, "}")
                self.call = None
            elif self.buf.strip() and not self.calls_seen:
                out.append(Event("content", CALLS_START + self.buf))
        elif self.buf:
            out.append(Event("reasoning" if self.state == "reasoning" else "content", self.buf))
        self.buf = ""
        return out


class GigaChat4(Dialect):
    name = "gigachat4"
    stop_texts = (MESSAGE_SEP, EOS)
    reasoning_close = "\n" + THINK_END + "\n\n"
    reasoning_wrap_up = "\n\nI have thought about this long enough; time to give my answer.\n" + THINK_END + "\n\n"
    forced_call_ends_reply = True
    nothink_lead = ""
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": False,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def template_kwargs(self, kwargs: dict) -> dict:
        out = dict(kwargs)
        out.setdefault("bos_token", BOS)
        if "preserve_reasoning" in out:
            out["drop_history_thinking"] = not out.pop("preserve_reasoning")
        if out.pop("enable_thinking", None) is False:
            out["reasoning"] = False
        out.pop("reasoning_effort", None)
        return out

    def template(self, source: str):
        return frontend.ChatTemplate(source)

    def parser(self, thinking: bool, tools: list[dict] | None, stream_tools: bool = True):
        return GigaChat4Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls) -> dict[str, str]:
        tags = {**frontend.literal_tags(controls), GCML_TOKEN: GCML_MARK}
        return dict(sorted(tags.items(), key=lambda t: -len(t[0])))

    def forced_opening(self, name: str | None) -> str:
        head = f"{CALLS_START}\n{INVOKE_START} name=\""
        return head + (f'{name}">\n' if name else "")
