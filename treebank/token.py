from typing import TypedDict, Required
import re

class TokenDict(TypedDict):
    id: Required[int]
    form: Required[str]
    lemma: Required[str]
    upos: Required[str]
    xpos: Required[str]
    feats: Required[dict[str, str]]
    head: Required[int]
    deprel: Required[str]
    deps: Required[str]
    miscs: Required[dict[str, str]]

class Token:

    pattern = re.compile(r"^\d+\t[^\t_]+\t_\t([A-Z]+)\t\1\t(_|[A-Z][a-z]+([A-Z][a-z]+)?=[A-Z][a-z]+(\|[A-Z][a-z]+([A-Z][a-z]+)?=[A-Z][a-z]+)*)\t\d+\t[a-z]+\t_\tSpaceAfter=(Yes|No)$")
    conllu_format = "{id}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{miscs}"

    def __init__(self, raw_conllu: str):
        assert self.pattern.fullmatch(raw_conllu), f"Wrong body format\n{raw_conllu}"
        id, form, lemma, upos, xpos, feats, head, deprel, deps, miscs = raw_conllu.rstrip('\n').split('\t')
        assert id != head, f"Self-head\n{raw_conllu}"
        self.id = int(id)
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = dict(feat.split('=') for feat in feats.split('|')) if feats != '_' else {}
        self.head = int(head)
        self.deprel = deprel
        self.deps = deps
        self.miscs = dict(misc.split('=') for misc in miscs.split('|')) if miscs != '_' else {}

    def __repr__(self):
        return f"<{type(self).__name__} {self.id}: {self.form}>"

    def to_conllu(self):
        return self.conllu_format.format(
            id=self.id,
            form=self.form,
            lemma=self.lemma,
            upos=self.upos,
            xpos=self.xpos,
            feats='|'.join(f"{k}={v}" for k, v in self.feats.items()) if self.feats else '_',
            head=self.head,
            deprel=self.deprel,
            deps=self.deps,
            miscs='|'.join(f"{k}={v}" for k, v in self.miscs.items()) if self.miscs else '_'
        )

    def to_dict(self):
        return TokenDict(
            id=self.id,
            form=self.form,
            lemma=self.lemma,
            upos=self.upos,
            xpos=self.xpos,
            feats=self.feats,
            head=self.head,
            deprel=self.deprel,
            deps=self.deps,
            miscs=self.miscs
        )
