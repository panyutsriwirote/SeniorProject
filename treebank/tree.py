from typing import overload, TypedDict, Required
import re
from .token import Token, TokenDict

class TreeDict(TypedDict):
    filename: Required[str]
    sent_id: Required[str]
    text: Required[str]
    tokens: Required[list[TokenDict]]

class Tree:

    header_pattern = re.compile(r"^# (?P<key>.+) = (?P<value>.+)$")
    conllu_format = "# filename = {filename}\n# sent_id = {sent_id}\n# text = {text}\n{body}"

    def __init__(self, raw_conllu: str):
        raw_lines = raw_conllu.split('\n')
        headers: dict[str, str] = {}
        for i, raw_line in enumerate(raw_lines):
            match = self.header_pattern.fullmatch(raw_line)
            if match:
                headers[match["key"]] = match["value"]
            else:
                raw_lines = raw_lines[i:]
                break
        assert "filename" in headers, f"Missing 'filename' header\n{headers}"
        assert "sent_id" in headers, f"Missing 'sent_id' header\n{headers}"
        assert "text" in headers, f"Missing 'text' header\n{headers}"
        self.filename = headers["filename"]
        self.sent_id = headers["sent_id"]
        self.text = headers["text"]
        self.__tokens: list[Token] = []
        found_root = False
        for i, raw_line in enumerate(raw_lines, start=1):
            token = Token(raw_line)
            assert token.id == i, f"Wrong token id in {self} at {token}"
            assert (token.head == 0) == (token.deprel == "root"), f"Inconsistent head, deprel in {self} at {token}"
            if token.head == 0:
                assert not found_root, f"Multiple root in {self}. Second root at {token}"
                found_root = True
            self.__tokens.append(token)
        assert found_root, f"Root not found in {self}"
        assert self[-1].miscs["SpaceAfter"] == "No", f"Last token in {self} has SpaceAfter=Yes"
        reconstructed_text = ''.join(token.form if token.miscs["SpaceAfter"] == "No" else token.form + ' ' for token in self)
        assert reconstructed_text == self.text, f"Text mismatch in {self}\nHeader: {self.text}\nActual: {reconstructed_text}"
        for start_token in self:
            token = start_token
            loop_counter = 0
            while token.head != 0:
                token = self[token.head-1]
                loop_counter += 1
                assert loop_counter < len(self), f"Loop in {self} starting at {start_token}"
        # Check for projectivity
        tree_is_projective = True
        for dep in self:
            if dep.deprel == "root":
                continue
            head = self[dep.head - 1]
            start, stop = (dep.id, head.id - 1) if dep.id < head.id else (head.id, dep.id - 1)
            arc_is_projective = True
            for token in self[start:stop]:
                token = self[token.head - 1]
                while token is not head:
                    if token.deprel == "root":
                        arc_is_projective = tree_is_projective = False
                        break
                    token = self[token.head - 1]
                if not arc_is_projective:
                    break
            dep.arc_is_projective = arc_is_projective
        self.is_projective = tree_is_projective
        self.num_non_projective_arcs = 0 if tree_is_projective else sum(not token.arc_is_projective for token in self)

    def __len__(self):
        return len(self.__tokens)

    def __iter__(self):
        return iter(self.__tokens)

    def __reversed__(self):
        return reversed(self.__tokens)

    @overload
    def __getitem__(self, index: int) -> Token: ...
    @overload
    def __getitem__(self, index: slice) -> list[Token]: ...
    def __getitem__(self, index: int | slice):
        return self.__tokens[index]

    def __repr__(self):
        return f"<{type(self).__name__} {self.filename}: {self.sent_id}>"

    def to_conllu(self):
        return self.conllu_format.format(
            filename=self.filename,
            sent_id=self.sent_id,
            text=self.text,
            body='\n'.join(token.to_conllu() for token in self)
        )

    def to_dict(self):
        return TreeDict(
            filename=self.filename,
            sent_id=self.sent_id,
            text=self.text,
            tokens=[token.to_dict() for token in self]
        )
