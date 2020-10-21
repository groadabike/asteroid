import os
import argparse
from pathlib import Path
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import ConvTasNet
from asteroid.data import DAMPVSEPDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import SingleSrcNegSTOI



# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py . 
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')


def main(conf):
    # Example of augmentation using Audiomentations
    # https://github.com/iver56/audiomentations
    # source_augmentations = Compose([
    #         Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0),
    #         FrequencyMask(min_frequency_band=0.3, max_frequency_band=0.5, p=0.5),
    #         TimeMask(min_band_part=0.2, max_band_part=0.5, fade=False, p=0.5),
    #         Normalize(p=1.0),
    #     ])

    source_augmentations = None
    # Define dataloader using ORIGINAL mixture.
    dataset_kwargs = {
        'root_path': Path(conf['data']['root_path']),
        'task': conf['data']['task'],
        'sample_rate': conf['data']['sample_rate'],
        'num_workers': conf['training']['num_workers'],
        'mixture': conf['data']['mixture']
    }

    train_set = DAMPVSEPDataset(
        split='train',
        random_segments=True,
        segment=conf['data']['segment'],
        samples_per_track=conf['data']['samples_per_track'],
        source_augmentations=source_augmentations,
        **dataset_kwargs
    )

    val_set = DAMPVSEPDataset(
        split='valid',
        **dataset_kwargs
    )

    train_loader = DataLoader(train_set, 
                              shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)

    val_loader = DataLoader(val_set, 
                            shuffle=False,
                            batch_size=1,
                            num_workers=conf['training']['num_workers'])

    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    model = ConvTasNet(**conf["filterbank"], **conf["masknet"])
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
  
    # Define scheduler
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                                      factor=0.5,
                                      patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)
    
    # Define Loss function.
    loss_func = CombineSTOIL1_Loss(alpha=conf['training']['loss_alpha'],
                                   sample_rate=conf['data']['sample_rate'])
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=10, verbose=True)
    
    early_stopping = False
    if conf["training"]["early_stop"]:
        early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=True)
                                  
    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None  
    trainer = pl.Trainer(
        max_epochs=conf['training']['epochs'],
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stopping,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend='ddp',
        train_percent_check=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


class CombineSTOIL1_Loss(torch.nn.Module):
    """
    Loss function combines L1 loss and STOI loss to focus the
    separation on the vocal segment. This has relevance specially
    when ORIGINAL mixture is selected.
    """

    def __init__(self, alpha=0.5, sample_rate=16000):
        super(CombineSTOIL1_Loss, self).__init__()
        self.alpha = alpha
        self.loss_vocal = SingleSrcNegSTOI(sample_rate=sample_rate, extended=False, use_vad=False)
        self.loss_background = torch.nn.L1Loss()

    def forward(self, est_targets, targets):
        l_vocal = self.loss_vocal(est_targets[:, 0, :], targets[:, 0, :])
        l_back = self.loss_background(est_targets[:, 1, :], targets[:, 1, :])

        loss = ((1 - self.alpha) * l_back) + (self.alpha * (torch.mean(l_vocal)))
        return loss


if __name__ == '__main__':
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
