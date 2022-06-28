import os


def download_ukb_wb_ld_matrix(target_dir='.'):
    """
    Download the LD matrix for the White British samples in the UK Biobank.
    :param target_dir: The path or directory where to store the LD matrix
    """

    import urllib.request
    from magenpy.utils.system_utils import makedir

    target_path = os.path.join(target_dir, "ukb_eur_50k_windowed_ld.tar.gz")

    try:
        makedir(os.path.dirname(target_path))
        urllib.request.urlretrieve("https://zenodo.org/record/6529229/files/ukb_eur_50k_windowed_ld.tar.gz?download=1",
                                   target_path)
        os.system(f"tar -xvzf {target_path}")
    except Exception as e:
        print(e)
