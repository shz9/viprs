
def download_ukb_wb_ld_matrix(target_dir='.', chromosome=None):
    """
    Download the LD matrix for the White British samples in the UK Biobank.
    :param target_dir: The path or directory where to store the LD matrix
    :param chromosome: An integer or list of integers with the chromosome numbers for which to download
    the LD matrices from Zenodo.
    """

    import urllib.request
    from magenpy.utils.system_utils import makedir
    from magenpy.utils.compute_utils import iterable
    import os
    from tqdm import tqdm

    if chromosome is None:
        chromosome = list(range(1, 23))
    elif not iterable(chromosome):
        chromosome = [chromosome]

    if len(chromosome) < 2:
        print("> Download LD matrix for chromosome", chromosome[0])

    for c in tqdm(chromosome, total=len(chromosome), disable=len(chromosome) < 2,
                  desc='Downloading LD matrices'):

        target_path = os.path.join(target_dir, f"chr_{c}.tar.gz")

        try:
            makedir(os.path.dirname(target_path))
            urllib.request.urlretrieve(f"https://zenodo.org/record/7036625/files/chr_{c}.tar.gz?download=1",
                                       target_path)
            os.system(f"tar -xvzf {target_path}")
        except Exception as e:
            print(e)
