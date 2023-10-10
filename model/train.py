from torch.optim import AdamW
from transformers import get_scheduler
from os import path
import torch
from treebank import TreeBank
from .transition_based import TransitionBasedModel
from .graph_based import GraphBasedModel

def train_model(
    *,
    model: TransitionBasedModel | GraphBasedModel,
    data_path: str,
    num_epochs: int,
    batch_size: int,
    save_path: str
):
    is_graph_based = isinstance(model, GraphBasedModel)

    train_set = TreeBank.from_conllu_file(path.join(data_path, "train.conllu"))
    dev_set = TreeBank.from_conllu_file(path.join(data_path, "dev.conllu"))
    test_set = TreeBank.from_conllu_file(path.join(data_path, "test.conllu"))

    optimizer = AdamW(
        params=model.parameters(),
        lr=3e-5,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    num_batches_per_epoch, remainder = divmod(len(train_set), batch_size)
    if remainder:
        num_batches_per_epoch += 1
    num_total_steps = num_epochs * num_batches_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_total_steps),
        num_training_steps=num_total_steps
    )

    max_las = 0.0
    for i in range(1, num_epochs + 1):
        model.train()
        print(f"EPOCH: {i}")
        start, stop = 0, batch_size
        batch = [tree for tree in train_set[start:stop] if is_graph_based or tree.is_projective]
        while batch:
            loss = model(batch).loss
            print(loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            start, stop = stop, stop + batch_size
            batch = [tree for tree in train_set[start:stop] if is_graph_based or tree.is_projective]
        model.eval()
        dev_metrics = model.evaluate(dev_set)
        print(f"DEV: {dev_metrics}")
        if dev_metrics["LAS"] > max_las:
            max_las = dev_metrics["LAS"]
            torch.save(model.state_dict(), save_path)
            print("Saved!")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_metrics = model.evaluate(test_set)
    print(f"TEST: {test_metrics}")
    return test_metrics
