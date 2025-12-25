"""
PWCG / IL-2 style .mission file reader + writer (text format) — UPDATED

Adds support for assignment values with colon-chains, e.g.:
  Time = 12:0:0;

Supports:
- Blocks:        Name { ... }
- Assignments:   Key = Value;
- Lists:         Key = [1,2,3];
- Tuple entries: 0 : 0;   or 500 : 90 : 5;
- Comments:      # ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import re


# ---------------------------
# AST types
# ---------------------------

@dataclass(frozen=True)
class Atom:
    """Unquoted token value as it appeared in the file (identifier or number-like)."""
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass
class ColonSeq:
    """Colon-separated value chain used as a single assignment value, e.g. 12:0:0"""
    parts: List["Value"]


Value = Union[str, Atom, List["Value"], ColonSeq]  # quoted string, unquoted atom, list, or colon-seq


@dataclass
class Comment:
    text: str  # includes leading '#'


@dataclass
class Assignment:
    key: Atom
    value: Value


@dataclass
class TupleEntry:
    parts: List[Value]  # values separated by ':' and ending with ';'


@dataclass
class Bare:
    value: Value  # a single value followed by ';'


@dataclass
class Block:
    name: Atom
    statements: List["Statement"]


Statement = Union[Comment, Assignment, TupleEntry, Bare, Block]


@dataclass
class MissionDoc:
    statements: List[Statement]


# ---------------------------
# Tokenizer
# ---------------------------

@dataclass(frozen=True)
class Token:
    kind: str  # 'WORD', 'STRING', 'SYMBOL', 'COMMENT', 'EOF'
    value: str
    pos: int


_SYMBOLS = {"{", "}", "[", "]", "=", ";", ":", ","}
_WORD_RE = re.compile(r"[^\s\{\}\[\]=;:,#\"']+")


def _tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        # comment to end-of-line
        if ch == "#":
            start = i
            while i < n and text[i] != "\n":
                i += 1
            tokens.append(Token("COMMENT", text[start:i], start))
            continue

        # string
        if ch == '"':
            start = i
            i += 1
            buf: List[str] = []
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    if nxt in {'"', "\\", "n", "t", "r"}:
                        buf.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
                        i += 2
                        continue
                if c == '"':
                    i += 1
                    break
                buf.append(c)
                i += 1
            else:
                raise ValueError(f"Unterminated string starting at {start}")
            tokens.append(Token("STRING", "".join(buf), start))
            continue

        # symbols
        if ch in _SYMBOLS:
            tokens.append(Token("SYMBOL", ch, i))
            i += 1
            continue

        # word
        m = _WORD_RE.match(text, i)
        if not m:
            raise ValueError(f"Unexpected character {ch!r} at position {i}")
        tokens.append(Token("WORD", m.group(0), i))
        i = m.end()

    tokens.append(Token("EOF", "", n))
    return tokens


# ---------------------------
# Parser
# ---------------------------

class _Parser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = list(tokens)
        self.idx = 0

    def _peek(self) -> Token:
        return self.tokens[self.idx]

    def _next(self) -> Token:
        tok = self.tokens[self.idx]
        self.idx = min(self.idx + 1, len(self.tokens) - 1)
        return tok

    def _expect(self, kind: str, value: Optional[str] = None) -> Token:
        tok = self._next()
        if tok.kind != kind:
            raise ValueError(f"Expected {kind}, got {tok.kind} at {tok.pos}")
        if value is not None and tok.value != value:
            raise ValueError(f"Expected {value!r}, got {tok.value!r} at {tok.pos}")
        return tok

    def parse_doc(self) -> MissionDoc:
        stmts = self._parse_statements(until_symbol=None)
        self._expect("EOF")
        return MissionDoc(statements=stmts)

    def _parse_statements(self, until_symbol: Optional[str]) -> List[Statement]:
        out: List[Statement] = []
        while True:
            tok = self._peek()

            if tok.kind == "EOF":
                if until_symbol is not None:
                    raise ValueError(f"Expected '{until_symbol}' before EOF")
                break

            if tok.kind == "SYMBOL" and until_symbol is not None and tok.value == until_symbol:
                break

            stmt = self._parse_statement()
            if stmt is not None:
                out.append(stmt)

        return out

    def _parse_statement(self) -> Optional[Statement]:
        tok = self._peek()

        if tok.kind == "COMMENT":
            self._next()
            return Comment(tok.value)

        first = self._parse_value_token_only()
        nxt = self._peek()

        # Block: Name { ... }
        if nxt.kind == "SYMBOL" and nxt.value == "{":
            if not isinstance(first, Atom):
                raise ValueError(f"Block name must be an unquoted token at {tok.pos}")
            self._expect("SYMBOL", "{")
            body = self._parse_statements(until_symbol="}")
            self._expect("SYMBOL", "}")
            return Block(name=first, statements=body)

        # Assignment: key = value(:value...)? ;
        if nxt.kind == "SYMBOL" and nxt.value == "=":
            if not isinstance(first, Atom):
                raise ValueError(f"Assignment key must be an unquoted token at {tok.pos}")
            self._expect("SYMBOL", "=")

            value: Value = self._parse_value()

            # NEW: allow colon-chained values in assignment (e.g. Time = 12:0:0;)
            if self._peek().kind == "SYMBOL" and self._peek().value == ":":
                parts: List[Value] = [value]
                while self._peek().kind == "SYMBOL" and self._peek().value == ":":
                    self._expect("SYMBOL", ":")
                    parts.append(self._parse_value())
                value = ColonSeq(parts=parts)

            self._expect("SYMBOL", ";")
            return Assignment(key=first, value=value)

        # Tuple entry: v : v (: v ... ) ;
        if nxt.kind == "SYMBOL" and nxt.value == ":":
            parts: List[Value] = [first]
            while self._peek().kind == "SYMBOL" and self._peek().value == ":":
                self._expect("SYMBOL", ":")
                parts.append(self._parse_value())
            self._expect("SYMBOL", ";")
            return TupleEntry(parts=parts)

        # Bare: value ;
        if nxt.kind == "SYMBOL" and nxt.value == ";":
            self._expect("SYMBOL", ";")
            return Bare(value=first)

        raise ValueError(f"Unexpected token sequence near {tok.pos}: {tok.kind} {tok.value!r}")

    def _parse_value_token_only(self) -> Value:
        tok = self._next()
        if tok.kind == "STRING":
            return tok.value
        if tok.kind == "WORD":
            return Atom(tok.value)
        if tok.kind == "SYMBOL" and tok.value == "[":
            items: List[Value] = []
            if self._peek().kind == "SYMBOL" and self._peek().value == "]":
                self._expect("SYMBOL", "]")
                return items
            while True:
                items.append(self._parse_value())
                if self._peek().kind == "SYMBOL" and self._peek().value == ",":
                    self._expect("SYMBOL", ",")
                    continue
                break
            self._expect("SYMBOL", "]")
            return items
        raise ValueError(f"Unexpected token {tok.kind} {tok.value!r} at {tok.pos}")

    def _parse_value(self) -> Value:
        tok = self._peek()
        if tok.kind in ("STRING", "WORD"):
            return self._parse_value_token_only()
        if tok.kind == "SYMBOL" and tok.value == "[":
            return self._parse_value_token_only()
        raise ValueError(f"Expected value at {tok.pos}, got {tok.kind} {tok.value!r}")


# ---------------------------
# Writer
# ---------------------------

def _escape_string(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\t", "\\t")
         .replace("\r", "\\r")
    )


def _value_to_text(v: Value) -> str:
    if isinstance(v, Atom):
        return v.text
    if isinstance(v, str):
        return f"\"{_escape_string(v)}\""
    if isinstance(v, ColonSeq):
        return ":".join(_value_to_text(p) for p in v.parts)
    if isinstance(v, list):
        return "[" + ",".join(_value_to_text(x) for x in v) + "]"
    return str(v)


def dumps(doc: MissionDoc, indent: str = "  ") -> str:
    lines: List[str] = []

    def emit_statement(stmt: Statement, level: int) -> None:
        pad = indent * level

        if isinstance(stmt, Comment):
            lines.append(stmt.text)
            return

        if isinstance(stmt, Assignment):
            lines.append(f"{pad}{stmt.key.text} = {_value_to_text(stmt.value)};")
            return

        if isinstance(stmt, TupleEntry):
            lines.append(f"{pad}" + " : ".join(_value_to_text(p) for p in stmt.parts) + ";")
            return

        if isinstance(stmt, Bare):
            lines.append(f"{pad}{_value_to_text(stmt.value)};")
            return

        if isinstance(stmt, Block):
            lines.append(f"{pad}{stmt.name.text}")
            lines.append(f"{pad}" + "{")
            for inner in stmt.statements:
                emit_statement(inner, level + 1)
            lines.append(f"{pad}" + "}")
            lines.append("")  # blank line between blocks
            return

        raise TypeError(f"Unknown statement type: {type(stmt)}")

    for s in doc.statements:
        emit_statement(s, 0)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def loads(text: str) -> MissionDoc:
    return _Parser(_tokenize(text)).parse_doc()


def read_mission_file(path: str, encoding: str = "utf-8") -> MissionDoc:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return loads(f.read())


def write_mission_file(doc: MissionDoc, path: str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding, newline="\n") as f:
        f.write(dumps(doc))


# ---------------------------
# Convenience builders
# ---------------------------

def A(text: str) -> Atom:
    return Atom(text)


def assign(key: str, value: Value) -> Assignment:
    return Assignment(key=A(key), value=value)


def block(name: str, *statements: Statement) -> Block:
    return Block(name=A(name), statements=list(statements))

def main():
    doc = read_mission_file("some.mission")
    write_mission_file(doc, "roundtrip.mission")
    

if __name__ == "__main__":
    raise SystemExit(main())

