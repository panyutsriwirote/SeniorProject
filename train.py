from treebank import TreeBank
from model import TransitionBasedModel
from torch.optim import AdamW

ThaiPUD = TreeBank.from_conllu_file("th_pud-ud-test.conllu")

model = TransitionBasedModel(
    "eager",
    ThaiPUD.tag_set,
    "airesearch/wangchanberta-base-att-spm-uncased"
)

optimizer = AdamW(
    params=model.parameters(),
    lr=3e-5,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999)
)

NUM_EPOCHS = 1
BATCH_SIZE = 8
for i in range(1, NUM_EPOCHS + 1):
    print(f"EPOCH: {i}")
    start, stop = 0, BATCH_SIZE
    batch = [tree for tree in ThaiPUD[start:stop] if tree.is_projective]
    while batch:
        loss = model(batch).loss
        print(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        start, stop = stop, stop + BATCH_SIZE
        batch = [tree for tree in ThaiPUD[start:stop] if tree.is_projective]
