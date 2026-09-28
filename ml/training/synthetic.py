import torch


def generate_batch(
    batch_size: int,
    seq_len: int,
    device,
):
    domain_ids = torch.randint(
        0,
        4,
        (batch_size,),
        device=device,
    )

    starts = torch.randint(
        0,
        16,
        (batch_size,),
        device=device,
    )

    steps = torch.tensor(
        [1, 2, 3, 5],
        device=device,
    )

    positions = torch.arange(
        seq_len + 1,
        device=device,
    )

    sequences = []

    for i in range(batch_size):
        domain = domain_ids[i]

        base = domain * 16
        step = steps[domain]

        seq = (
            starts[i]
            + step * positions
        ) % 16

        seq = seq + base

        sequences.append(seq)

    sequences = torch.stack(sequences).long()

    input_ids = sequences[:, :-1]
    labels = sequences[:, 1:]

    return input_ids, labels, domain_ids
