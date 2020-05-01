def list_collate(batch):
    inputs, targets = map(list, zip(*batch))
    return (inputs, targets), None


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    ds = [
        (torch.rand(3, 3, 4),
         {
            'image_id': [0],
            'area': 1.5,
            'iscrowd': True,
        }),
        (torch.rand(3, 1, 6),
         {
            'image_id': [1],
            'area': 1.7,
            'iscrowd': False,
        }),
        (torch.rand(3, 5, 2),
         {
            'image_id': [2],
            'area': 3.2,
            'iscrowd': True,
        }),
    ]
    dl = DataLoader(ds, batch_size=2,
                    collate_fn=list_collate)

    for i, (inps, targets) in enumerate(dl):
        print(i)
        print(inps)
        print(targets)
