from torch.nn import Module, Linear, Dropout, CrossEntropyLoss, Parameter
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
import torch
from treebank.tree import Tree, Relation
from treebank import TreeBank
from spanningtrees import MST

@dataclass
class GraphBasedModelOutput:
    arc_scores: list[Tensor]
    label_scores: list[Tensor]
    loss: Tensor

class FFNN(Module):

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.dropout1 = Dropout(dropout)
        self.dense1 = Linear(in_features, out_features)
        self.dropout2 = Dropout(dropout)
        self.dense2 = Linear(out_features, out_features)

    def forward(self, x: Tensor):
        x = self.dropout1(x)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return torch.relu(x)

class Biaffine(Module):

    def __init__(self, hidden_size: int, initializer_range: float, num_labels: int | None = None):
        super().__init__()
        if num_labels is None:
            self.U = Parameter(torch.zeros(hidden_size, hidden_size))
        else:
            self.U = Parameter(torch.zeros(num_labels, hidden_size, hidden_size))
        # self.W = Parameter(torch.zeros(hidden_size, hidden_size))
        # self.b = Parameter(torch.zeros(1))
        self.U.data.normal_(mean=0.0, std=initializer_range)
        # self.W.data.normal_(mean=0.0, std=initializer_range)
        # self.b.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, head: Tensor, dep: Tensor):
        return head @ self.U @ dep.transpose(0, 1)

class GraphBasedModel(Module):

    def __init__(
        self,
        tag_set: list[str],
        transformer_model: str,
        space_token: str = "<_>"
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer = AutoModel.from_pretrained(transformer_model, add_pooling_layer=False)

        self.id_to_label = dict(enumerate(tag_set))
        self.label_to_id = {label: i for i, label in enumerate(tag_set)}
        self.space_token = space_token
        vocab = self.tokenizer.get_vocab()
        self.space_ids = {vocab.get(space_token), vocab.get('▁')}
        self.space_ids.discard(None)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        config = self.transformer.config
        hidden_size = config.hidden_size
        initializer_range = config.initializer_range
        # ROOT embeddings
        self.root_embedding = Parameter(torch.zeros(1, hidden_size))
        self.root_embedding.data.normal_(mean=0.0, std=initializer_range)
        # FFNN
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.arc_head = FFNN(hidden_size, hidden_size, classifier_dropout)
        self.arc_dep = FFNN(hidden_size, hidden_size, classifier_dropout)
        self.label_head = FFNN(hidden_size, hidden_size, classifier_dropout)
        self.label_dep = FFNN(hidden_size, hidden_size, classifier_dropout)
        # Biaffine layers
        self.arc_biaffine = Biaffine(hidden_size, initializer_range)
        self.label_biaffine = Biaffine(hidden_size, initializer_range, len(self.id_to_label))
        # Loss
        self.loss_func = CrossEntropyLoss()

        self.to(self.device)

    def __encode_and_pool(self, trees: list[Tree]):
        # Create list of lists of strings to be tokenized
        words: list[list[str]] = [[] for _ in trees]
        for tree, word in zip(trees, words):
            for token in tree:
                word.append(token.form)
                if self.space_token != ' ' and token.miscs["SpaceAfter"] == "Yes":
                    word.append(self.space_token)
        # Tokenize
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        # Encode using transformer
        encoded = self.transformer(**tokenized).last_hidden_state
        # Select the encoding of the first non-space token of each word
        select_indice: list[list[int]] = [[] for _ in trees]
        for i, (select_index, input_ids) in enumerate(zip(select_indice, tokenized.input_ids)):
            word_ids = tokenized.word_ids(batch_index=i)
            last_word_id = None
            for j, (word_id, token_id) in enumerate(zip(word_ids, input_ids)):
                if word_id in (None, last_word_id) or token_id.item() in self.space_ids:
                    continue
                select_index.append(j)
                last_word_id = word_id
        return [
            torch.cat([
                self.root_embedding,
                encoded[i, select_index]
            ])
            for i, select_index in enumerate(select_indice)
        ]

    def forward(self, trees: list[Tree]):
        pooled_encodings = self.__encode_and_pool(trees)
        arc_scores: list[Tensor] = []
        label_scores: list[Tensor] = []
        losses: list[Tensor] = []
        for tree, encoding in zip(trees, pooled_encodings):
            # FFNN
            arc_head = self.arc_head(encoding)
            arc_dep = self.arc_dep(encoding)
            label_head = self.label_head(encoding)
            label_dep = self.label_dep(encoding)
            # Biaffine
            arc_score = self.arc_biaffine(arc_head, arc_dep)
            label_score = self.label_biaffine(label_head, label_dep)
            arc_scores.append(arc_score)
            label_scores.append(label_score)
            # Loss
            arc_x = arc_score[:,1:].transpose(0, 1)
            arc_y = torch.tensor([token.head for token in tree]).to(self.device)
            arc_loss = self.loss_func(arc_x, arc_y)
            label_x = torch.stack([label_score[:,token.head,token.id] for token in tree])
            label_y = torch.tensor([self.label_to_id[token.deprel] for token in tree]).to(self.device)
            label_loss = self.loss_func(label_x, label_y)
            losses.append(arc_loss + label_loss)
        loss = torch.stack(losses).mean()
        return GraphBasedModelOutput(
            arc_scores=arc_scores,
            label_scores=label_scores,
            loss=loss
        )

    def evaluate(self, test_treebank: TreeBank):
        num_head_correct = 0
        num_head_label_correct = 0
        num_total = 0
        for tree in test_treebank:
            num_total += len(tree)
            result = self.parse(tree)
            for relation in result:
                if relation.dep.head_token is relation.head:
                    num_head_correct += 1
                    if relation.dep.deprel == relation.deprel:
                        num_head_label_correct += 1
        return {"UAS": num_head_correct / num_total, "LAS": num_head_label_correct / num_total}

    @torch.no_grad()
    def parse(self, tree: Tree):
        encoding = self.__encode_and_pool([tree])[0]
        # FFNN
        arc_head = self.arc_head(encoding)
        arc_dep = self.arc_dep(encoding)
        label_head = self.label_head(encoding)
        label_dep = self.label_dep(encoding)
        # Biaffine
        arc_score = self.arc_biaffine(arc_head, arc_dep)
        label_score = self.label_biaffine(label_head, label_dep)
        # Decode
        relations: list[Relation] = []
        tokens = [Tree._ROOT] + list(tree)
        head_ids = MST(arc_score.to("cpu")).mst(True)[1:]
        for token_id, head_id in enumerate(head_ids, start=1):
            label_pred = self.id_to_label[label_score[:,head_id,token_id].argmax().item()]
            relations.append(
                Relation(
                    head=tokens[head_id],
                    dep=tokens[token_id],
                    deprel=label_pred
                )
            )
        return relations
