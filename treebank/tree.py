from typing import overload, TypedDict, Required
import re
from .token import Token, TokenDict

class TreeDict(TypedDict):
    filename: Required[str]
    sent_id: Required[str]
    text: Required[str]
    tokens: Required[list[TokenDict]]

class Tree:

    first_header_pattern = re.compile(r"^# filename = ([^\t_]+)$")
    second_header_pattern = re.compile(r"^# sent_id = (\d+(\.\d+)?)$")
    third_header_pattern = re.compile(r"^# text = ([^\t_]+)$")
    conllu_format = "# filename = {filename}\n# sent_id = {sent_id}\n# text = {text}\n{body}"

    def __init__(self, raw_conllu: str):
        first_header, second_header, third_header, *raw_lines = raw_conllu.split('\n')
        assert (match := Tree.first_header_pattern.fullmatch(first_header)), f"Wrong first header format\n{first_header}"
        self.filename: str = match[1]
        assert (match := Tree.second_header_pattern.fullmatch(second_header)), f"Wrong second header format\n{second_header}"
        self.sent_id: str = match[1]
        assert (match := Tree.third_header_pattern.fullmatch(third_header)), f"Wrong third header format\n{third_header}"
        self.text: str = match[1]
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
        return Tree.conllu_format.format(
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
