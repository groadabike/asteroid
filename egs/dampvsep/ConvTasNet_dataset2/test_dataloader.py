from asteroid.data import DAMPVSEPSinglesDatasetSegmented
from pathlib import Path
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dataset_kwargs = {
        "root_path": Path('/media/gerardo/TOSHIBA/DataSets/DAMP/DAMP-VSEP'),
        "sample_rate": 16000,
        "task": "separation",
        "mixture": 'remix',
    }

    valid_set = DAMPVSEPSinglesDatasetSegmented(
        split='valid',
        segment=3.0,
        **dataset_kwargs,
    )

    train_set = DAMPVSEPSinglesDatasetSegmented(
        split='train_english',
        segment=3.0,
        **dataset_kwargs,
    )

    print(len(valid_set))
    print(len(train_set))
    valid_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=12)
    for x, y in valid_loader:
        print(x.shape, y.shape)
