import gzip
import os
import shutil
import tempfile
from numbers import Integral
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

PGS_CATALOG_METADATA_DEFAULTS = {
    "pgs_id": "PGS000000",
    "pgs_name": "VIPRS",
    "trait_reported": "Not reported",
    "trait_mapped": "Not reported",
    "trait_efo": "EFO_0000000",
    "weight_type": "beta",
    "pgp_id": "PGP000000",
    "citation": "Not reported",
    "license": "EMBL-EBI Terms of Use",
}
PGS_CATALOG_METADATA_FIELDS = tuple(PGS_CATALOG_METADATA_DEFAULTS) + ("genome_build",)
PGS_CATALOG_COLUMN_MAP = {
    "rsID": "SNP",
    "chr_name": "CHR",
    "chr_position": "POS",
    "effect_allele": "A1",
    "other_allele": "A2",
    "effect_weight": "BETA",
}


def parse_pgs_catalog_metadata(metadata=None):
    """Parse metadata supplied as a mapping or comma-separated ``key=value`` string."""

    if metadata is None:
        return {}
    if not isinstance(metadata, str):
        return metadata.copy()

    parsed_metadata = {}
    for item in metadata.split(","):
        if "=" not in item:
            raise ValueError(
                "PGS Catalog metadata must be comma-separated key=value pairs."
            )
        key, value = (part.strip() for part in item.split("=", 1))
        if not key or not value:
            raise ValueError("PGS Catalog metadata keys and values cannot be empty.")
        if key in parsed_metadata:
            raise ValueError(f"Duplicate PGS Catalog metadata field: {key}")
        parsed_metadata[key] = value

    return parsed_metadata


def prepare_pgs_catalog_metadata(metadata=None):
    """Apply defaults and validate PGS Catalog scoring-file metadata."""

    metadata = parse_pgs_catalog_metadata(metadata)
    unknown_metadata = set(metadata) - set(PGS_CATALOG_METADATA_FIELDS)
    if unknown_metadata:
        raise ValueError(
            f"Unsupported PGS Catalog metadata: {sorted(unknown_metadata)}"
        )

    prepared_metadata = PGS_CATALOG_METADATA_DEFAULTS | metadata
    if prepared_metadata.get("genome_build") not in ("GRCh37", "GRCh38"):
        raise ValueError(
            "PGS Catalog output requires `genome_build` to be either "
            "'GRCh37' or 'GRCh38'."
        )

    for key, value in prepared_metadata.items():
        if value is None or any(char in str(value) for char in ("\n", "\r", "\t")):
            raise ValueError(f"Invalid value for PGS Catalog metadata field `{key}`.")

    return prepared_metadata


def write_pgs_catalog_scoring_file(
    parameter_table, f_name, metadata=None, effect_column="BETA"
):
    """
    Write inferred effect sizes using PGS Catalog scoring-file format 2.0.

    :param parameter_table: A pandas DataFrame containing VIPRS variant parameters.
    :param f_name: Output filename. A bare prefix receives the ``.txt.gz`` suffix.
    :param metadata: PGS Catalog metadata. ``genome_build`` is required; defaults are
        used for metadata that may not be known until catalog submission.
    :param effect_column: Parameter-table column to write as ``effect_weight``.
    :return: The path of the scoring file that was written.
    """

    metadata = prepare_pgs_catalog_metadata(metadata)

    required_columns = {"SNP", "CHR", "POS", "A1", "A2", effect_column}
    missing_columns = required_columns - set(parameter_table.columns)
    if missing_columns:
        raise ValueError(
            "Cannot create PGS Catalog output; missing columns: "
            f"{sorted(missing_columns)}"
        )

    table = parameter_table[["SNP", "CHR", "POS", "A1", "A2", effect_column]].rename(
        columns={
            "SNP": "rsID",
            "CHR": "chr_name",
            "POS": "chr_position",
            "A1": "effect_allele",
            "A2": "other_allele",
            effect_column: "effect_weight",
        }
    )

    required_values = [
        "chr_name",
        "chr_position",
        "effect_allele",
        "other_allele",
        "effect_weight",
    ]
    if table[required_values].isna().any().any():
        raise ValueError(
            "PGS Catalog output cannot contain missing chromosome, position, "
            "allele, or effect-weight values."
        )
    if not np.isfinite(pd.to_numeric(table["effect_weight"], errors="coerce")).all():
        raise ValueError("PGS Catalog effect weights must be finite numeric values.")

    header = [
        "###PGS CATALOG SCORING FILE - see "
        "https://www.pgscatalog.org/downloads/#dl_ftp_scoring for additional information",
        "#format_version=2.0",
        "##POLYGENIC SCORE (PGS) INFORMATION",
        f"#pgs_id={metadata['pgs_id']}",
        f"#pgs_name={metadata['pgs_name']}",
        f"#trait_reported={metadata['trait_reported']}",
        f"#trait_mapped={metadata['trait_mapped']}",
        f"#trait_efo={metadata['trait_efo']}",
        f"#genome_build={metadata['genome_build']}",
        f"#variants_number={len(table)}",
        f"#weight_type={metadata['weight_type']}",
        "##SOURCE INFORMATION",
        f"#pgp_id={metadata['pgp_id']}",
        f"#citation={metadata['citation']}",
        f"#license={metadata['license']}",
    ]

    f_name = str(f_name)
    if not f_name.endswith((".txt", ".txt.gz")):
        f_name += ".txt.gz"

    open_file = gzip.open if f_name.endswith(".gz") else open
    with open_file(f_name, mode="wt", encoding="utf-8", newline="") as output_file:
        output_file.write("\n".join(header) + "\n")
        table.to_csv(output_file, sep="\t", index=False)

    return f_name


def read_pgs_catalog_scoring_file(f_name):
    """Read a PGS Catalog scoring file into VIPRS parameter-table format."""

    table = pd.read_csv(f_name, sep="\t", comment="#")
    required_columns = {
        "chr_name",
        "chr_position",
        "effect_allele",
        "other_allele",
        "effect_weight",
    }
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            "Cannot read PGS Catalog scoring file; missing columns: "
            f"{sorted(missing_columns)}"
        )

    table = table.rename(
        columns={
            column: PGS_CATALOG_COLUMN_MAP[column]
            for column in PGS_CATALOG_COLUMN_MAP
            if column in table.columns
        }
    )
    if table[["CHR", "POS", "A1", "A2", "BETA"]].isna().any().any():
        raise ValueError(
            "PGS Catalog scoring files cannot contain missing chromosome, "
            "position, allele, or effect-weight values."
        )

    table["BETA"] = pd.to_numeric(table["BETA"], errors="raise")
    if not np.isfinite(table["BETA"]).all():
        raise ValueError("PGS Catalog effect weights must be finite numeric values.")

    return table


def download_ld_matrix(
    target_dir=".", chromosome=None, *, ancestry, source="huggingface",
    panel="ukb", overwrite=False,
):
    """Download hosted LD archives locally without extracting them.

    :param target_dir: Destination root. Files are saved under
        ``<target_dir>/<panel>/<ancestry>/`` to keep reference panels separate.
    :param chromosome: An autosome number or iterable of autosome numbers (1–22).
        ``None`` downloads all 22 chromosomes from Hugging Face. Zenodo only
        provides whole-ancestry archives, so requires ``None``.
    :param ancestry: One of EUR, CSA, AFR, EAS, MID, or AMR (case insensitive).
    :param source: ``"huggingface"`` (default) or ``"zenodo"``.
    :param panel: ``"ukb"`` (UK Biobank, GRCh37/hg19) or ``"hgdp-1kg"``
        (HGDP + 1000 Genomes, GRCh38/hg38). Match the summary-statistics build.
    :param overwrite: Replace existing archives if True. Otherwise reuse them.
    :return: A list of pathlib.Path objects for the downloaded or reused archives.
        Hugging Face ZIPs can be passed directly to magenpy; Zenodo tar.gz files
        must be extracted before use.

    Downloads stream to temporary files in the destination directory and are
    renamed only after completion. A failed download leaves any existing archive
    intact; successfully downloaded chromosomes are retained for a later retry.
    When installed, ``huggingface_hub.hf_hub_download`` handles Hugging Face
    transfers, using a temporary local directory rather than duplicating large
    archives in the global Hub cache. Otherwise, HTTP streaming is used.
    Hub download failures are reported directly, without retrying via HTTP.
    No Hugging Face package or authentication is required for these public files.
    """
    panels = {"ukb": ("shz9/ukb-ld", "14614207"),
              "hgdp-1kg": ("shz9/hgdp-1kg-ld", "24147521")}
    ancestry = str(ancestry).upper()
    source = str(source).lower()
    panel = str(panel).lower()
    if ancestry not in {"EUR", "CSA", "AFR", "EAS", "MID", "AMR"}:
        raise ValueError("Ancestry must be one of EUR, CSA, AFR, EAS, MID, or AMR.")
    if source not in {"huggingface", "zenodo"}:
        raise ValueError("Download source must be 'huggingface' or 'zenodo'.")
    if panel not in panels:
        raise ValueError("LD panel must be 'ukb' (hg19) or 'hgdp-1kg' (hg38).")
    if source == "zenodo" and chromosome is not None:
        raise ValueError(
            "Zenodo provides whole-ancestry archives only. Set chromosome=None "
            "to download the full archive, or use source='huggingface' to select chromosomes."
        )

    if chromosome is None:
        chromosomes = list(range(1, 23))
    elif isinstance(chromosome, Integral) and not isinstance(chromosome, bool):
        chromosomes = [int(chromosome)]
    else:
        try:
            chromosomes = list(chromosome)
        except TypeError as error:
            raise ValueError("Chromosome must be an integer or iterable of integers from 1 to 22.") from error
    if not chromosomes or any(
        isinstance(c, bool) or not isinstance(c, Integral) or not 1 <= c <= 22
        for c in chromosomes
    ):
        raise ValueError("Chromosomes must be integers from 1 to 22; select at least one.")
    chromosomes = list(dict.fromkeys(int(c) for c in chromosomes))

    repository, record_id = panels[panel]
    if source == "huggingface":
        downloads = [
            (f"chr_{c}.zip", f"https://huggingface.co/datasets/{repository}/resolve/main/{ancestry}/chr_{c}.zip")
            for c in chromosomes
        ]
    else:
        downloads = [(f"{ancestry}.tar.gz",
                      f"https://zenodo.org/records/{record_id}/files/{ancestry}.tar.gz?download=1")]

    hf_download = None
    if source == "huggingface":
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as error:
            if error.name != "huggingface_hub":
                raise
        else:
            hf_download = hf_hub_download

    destination = Path(target_dir) / panel / ancestry
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in downloads:
        path = destination / filename
        if path.is_file() and not overwrite:
            if path.stat().st_size == 0:
                raise ValueError(f"Existing archive is empty: {path}. Use overwrite=True to replace it.")
            paths.append(path)
            continue
        temporary = None
        try:
            if hf_download is not None:
                # Stage on the destination filesystem so completion can use an
                # atomic rename, including when overwriting an existing archive.
                with tempfile.TemporaryDirectory(dir=destination, prefix=f".{filename}.") as staging:
                    downloaded = Path(hf_download(
                        repo_id=repository, repo_type="dataset", revision="main",
                        filename=f"{ancestry}/{filename}", local_dir=staging,
                        force_download=overwrite,
                    ))
                    if downloaded.stat().st_size == 0:
                        raise OSError("The download was empty.")
                    os.replace(downloaded, path)
                paths.append(path)
                continue
            with urlopen(url, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=destination, prefix=f".{filename}.", suffix=".part", delete=False) as output:
                    temporary = Path(output.name)
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                    downloaded_bytes = output.tell()
                expected_bytes = response.headers.get("Content-Length")
                if downloaded_bytes == 0 or (expected_bytes is not None and downloaded_bytes != int(expected_bytes)):
                    raise OSError("The download was empty or incomplete.")
            os.replace(temporary, path)
        except Exception as error:
            raise OSError(
                f"Could not download {panel}/{ancestry}/{filename} from {source}: {error}"
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        paths.append(path)
    return paths
