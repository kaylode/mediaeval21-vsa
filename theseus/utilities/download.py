import gdown
import os
import os.path as osp
import urllib.request as urlreq
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

def download_from_drive(id_or_url, output, md5=None, quiet=False, cache=True):
    if id_or_url.startswith('http') or id_or_url.startswith('https'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    if not cache:
        return gdown.download(url, output, quiet=quiet)
    else:
        return gdown.cached_download(url, md5=md5, quiet=quiet)

def download_from_wandb(filename, run_path, save_dir):
    import wandb
    try:
        path = wandb.restore(
            filename, run_path=run_path, root=save_dir)
        return path.name
    except:
        LOGGER.text("Failed to download from wandb.",
                level=LoggerObserver.ERROR)
        return None