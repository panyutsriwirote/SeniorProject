from torch.nn import Module, Linear, Dropout, CrossEntropyLoss
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import Literal
from dataclasses import dataclass
import torch
from treebank.tree import Tree, Relation
from treebank.token import Token

@dataclass
class TransitionBasedModelOutput:
    output: Tensor
    loss: Tensor

class TransitionBasedModel(Module):

    def __init__(
        self,
        action_set: Literal["standard", "eager"],
        tag_set: list[str],
        transformer_path: str,
        space_token: str = "<_>"
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
        self.model = AutoModel.from_pretrained(transformer_path, add_pooling_layer=False)

        if action_set not in ("standard", "eager"):
            raise ValueError(f"Invalid action set: {action_set}")
        self.action_set = action_set
        tag_set = ["Shift"] + [f"LeftArc-{tag}" for tag in tag_set] + [f"RightArc-{tag}" for tag in tag_set]
        if action_set == "eager":
            tag_set.append("Reduce")
        self.id_to_label = dict(enumerate(tag_set))
        self.label_to_id = {label: i for i, label in enumerate(tag_set)}
        self.space_token = space_token

        # Classifier
        config = self.model.config
        feature_size = config.hidden_size * 2
        self.dense = Linear(feature_size, feature_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.out_proj = Linear(feature_size, len(self.id_to_label))

    def __encode_and_pool(self, trees: list[Tree]):
        # Create list of lists of strings to be tokenized
        words: list[list[str]] = [[] for _ in trees]
        for tree, word in zip(trees, words):
            for token in tree:
                word.append(token.form)
                if token.miscs["SpaceAfter"] == "Yes":
                    word.append(self.space_token)
        # Tokenize
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            return_tensors="pt"
        )
        # Encode using transformer
        encoded = self.model(**tokenized).last_hidden_state
        # Select the encoding of the first token of each word
        # Plus the encoding of the first token of the sentence (ROOT)
        select_indice: list[list[int]] = [[0] for _ in trees]
        for i, (tree, select_index) in enumerate(zip(trees, select_indice)):
            word_ids = tokenized.word_ids(batch_index=i)
            token_iter = iter(tree)
            is_space = False
            last_word_id = None
            for j, word_id in enumerate(word_ids):
                if word_id is None or word_id == last_word_id:
                    continue
                if is_space:
                    is_space = False
                    continue
                select_index.append(j)
                last_word_id = word_id
                try:
                    is_space = next(token_iter).miscs["SpaceAfter"] == "Yes"
                except StopIteration:
                    break
        return [
            encoded[i, select_index]
            for i, select_index in enumerate(select_indice)
        ]

    def __classifier(self, x: Tensor):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

    def forward(self, trees: list[Tree]):
        # Encode and pool
        pooled_encodings = self.__encode_and_pool(trees)
        # Create training samples
        inputs: list[Tensor] = []
        labels: list[int] = []
        if self.action_set == "standard":
            for tree, encoding in zip(trees, pooled_encodings):
                for state, action in tree.to_transitions("standard"):
                    if len(state.stack) < 2:
                        continue
                    inputs.append(torch.cat([
                        encoding[state.stack[-2].id],
                        encoding[state.stack[-1].id]
                    ]))
                    labels.append(self.label_to_id[action])
        elif self.action_set == "eager":
            for tree, encoding in zip(trees, pooled_encodings):
                for state, action in tree.to_transitions("eager"):
                    if len(state.buffer) < 1:
                        continue
                    inputs.append(torch.cat([
                        encoding[state.stack[-1].id],
                        encoding[state.buffer[0].id]
                    ]))
                    labels.append(self.label_to_id[action])
        x = torch.stack(inputs)
        y = torch.tensor(labels)
        # Classifier
        x = self.__classifier(x)
        # Loss
        loss_func = CrossEntropyLoss()
        loss = loss_func(x, y)
        # Return
        return TransitionBasedModelOutput(
            output=x,
            loss=loss
        )

    @torch.no_grad()
    def parse(self, tree: Tree, trace: bool = False):
        encoding = self.__encode_and_pool([tree])[0]
        stack = [Tree._ROOT]
        buffer = list(tree)
        relations: list[Relation] = []
        if trace:
            show_state(stack, buffer)
        if self.action_set == "standard":
            while len(stack) > 1 or buffer:
                if len(stack) < 2:
                    stack.append(buffer.pop(0))
                else:
                    top = stack[-1]
                    second = stack[-2]
                    x = torch.stack([torch.cat([
                        encoding[second.id],
                        encoding[top.id]
                    ])])
                    pred = self.__classifier(x)[0]
                    action_ranks = map(
                        self.id_to_label.__getitem__,
                        sorted(
                            range(len(self.id_to_label)),
                            key=lambda i: pred[i],
                            reverse=True
                        )
                    )
                    for action in action_ranks:
                        if action == "Shift":
                            if buffer:
                                stack.append(buffer.pop(0))
                                break
                            else:
                                continue
                        arc, rel = action.split("-")
                        if arc == "LeftArc":
                            if second is not Tree._ROOT:
                                relations.append(
                                    Relation(
                                        head=top,
                                        dep=second,
                                        deprel=rel
                                    )
                                )
                                stack.pop(-2)
                            else:
                                continue
                        elif arc == "RightArc":
                            relations.append(
                                Relation(
                                    head=second,
                                    dep=top,
                                    deprel=rel
                                )
                            )
                            stack.pop()
                        break
                if trace:
                    show_state(stack, buffer)
        elif self.action_set == "eager":
            while len(stack) > 1 or buffer:
                if not buffer:
                    stack.pop()
                else:
                    top = stack[-1]
                    front = buffer[0]
                    x = torch.stack([torch.cat([
                        encoding[top.id],
                        encoding[front.id]
                    ])])
                    pred = self.__classifier(x)[0]
                    action_ranks = map(
                        self.id_to_label.__getitem__,
                        sorted(
                            range(len(self.id_to_label)),
                            key=lambda i: pred[i],
                            reverse=True
                        )
                    )
                    for action in action_ranks:
                        if action == "Reduce":
                            if (
                                len(stack) > 1 and
                                any(relation.dep is top for relation in relations)
                            ):
                                stack.pop()
                                break
                            else:
                                continue
                        if action == "Shift":
                            stack.append(buffer.pop(0))
                            break
                        arc, rel = action.split("-")
                        if (
                            arc == "LeftArc" and
                            not any(relation.dep is top for relation in relations)
                        ):
                            relations.append(
                                Relation(
                                    head=front,
                                    dep=top,
                                    deprel=rel
                                )
                            )
                            stack.pop()
                        elif arc == "RightArc":
                            relations.append(
                                Relation(
                                    head=top,
                                    dep=front,
                                    deprel=rel
                                )
                            )
                            stack.append(buffer.pop(0))
                        break
                if trace:
                    show_state(stack, buffer)
        return relations

def show_state(stack: list[Token], buffer: list[Token]):
    print(f"[{', '.join(token.form for token in stack)}], [{', '.join(token.form for token in buffer)}]")
