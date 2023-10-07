from treebank import TreeBank
from model import TransitionBasedModel
from torch.optim import AdamW
from transformers import get_scheduler
import torch

train_set = TreeBank.from_conllu_file("data/thai_pud/train.conllu")
dev_set = TreeBank.from_conllu_file("data/thai_pud/dev.conllu")
test_set = TreeBank.from_conllu_file("data/thai_pud/test.conllu")

model = TransitionBasedModel(
    "eager",
    train_set.tag_set,
    "airesearch/wangchanberta-base-att-spm-uncased"
)

optimizer = AdamW(
    params=model.parameters(),
    lr=3e-5,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999)
)

NUM_EPOCHS = 10
BATCH_SIZE = 8

num_batches_per_epoch, remainder = divmod(len(train_set), BATCH_SIZE)
if remainder:
    num_batches_per_epoch += 1
num_total_steps = NUM_EPOCHS * num_batches_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_total_steps),
    num_training_steps=num_total_steps
)

max_las = 0.0
for i in range(1, NUM_EPOCHS + 1):
    model.train()
    print(f"EPOCH: {i}")
    start, stop = 0, BATCH_SIZE
    batch = [tree for tree in train_set[start:stop] if tree.is_projective]
    while batch:
        loss = model(batch).loss
        print(loss.item())
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        start, stop = stop, stop + BATCH_SIZE
        batch = [tree for tree in train_set[start:stop] if tree.is_projective]
    model.eval()
    las = model.evaluate(dev_set)["LAS"]
    print(f"LAS: {las}")
    if las > max_las:
        max_las = las
        torch.save(model, "model.pt")
        print("Saved!")
model = torch.load("model.pt")
model.eval()
print(f"TEST: {model.evaluate(test_set)}")
