from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
from treebank.tree import Tree

class GraphBasedModel(Module):

    def __init__(self, transformer_model: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.model = AutoModel.from_pretrained(transformer_model, add_pooling_layer=False)

    def forward(self, tree: Tree):
        pass
