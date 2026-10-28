import asyncio as _asyncio
import gzip
import html as _html
import re as _re
from urllib.parse import unquote as _unquote
from urllib.parse import urljoin as _urljoin
from urllib.parse import urlsplit as _urlsplit

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
GWAS_CATALOG_SUMSTATS_ROOT = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
)


def is_gwas_catalog_accession(value):
    """Return whether ``value`` is a GWAS Catalog study accession."""

    return isinstance(value, str) and _re.fullmatch(
        r"GCST\d+", value.strip(), _re.IGNORECASE
    ) is not None


def gwas_catalog_directory(accession):
    """Return the GWAS Catalog summary-statistics directory for an accession."""

    match = _re.fullmatch(r"GCST(\d+)", str(accession).strip(), _re.IGNORECASE)
    if match is None:
        raise ValueError("Enter a GWAS Catalog accession such as GCST005195.")

    number = int(match.group(1))
    if not 0 < number <= 2**53 - 1:
        raise ValueError("The GWAS Catalog accession is out of range.")

    width = max(6, len(match.group(1)))
    start = (number - 1) // 1000 * 1000 + 1
    study_id = f"GCST{number:0{width}d}"
    study_range = f"GCST{start:0{width}d}-GCST{start + 999:0{width}d}"
    return f"{GWAS_CATALOG_SUMSTATS_ROOT}{study_range}/{study_id}/"


def _listing_links(listing, directory):
    origin = _urlsplit(directory)[:2]
    links = (
        _urljoin(directory, _html.unescape(match.group(1)))
        for match in _re.finditer(
            r"href=[\"']([^\"']+)[\"']", listing, _re.IGNORECASE
        )
    )
    return [
        url
        for url in links
        if _urlsplit(url)[:2] == origin and url.startswith(directory)
    ]


def _is_build_37_filename(url):
    return _re.search(
        r"(?:grch37|build(?:grch)?37|hg19)",
        _unquote(_urlsplit(url).path),
        _re.IGNORECASE,
    ) is not None


def _metadata_is_build_37(metadata):
    match = _re.search(
        r"(?:^|\n)\s*genome_assembly\s*:\s*[\"']?([^\s#\"']+)",
        metadata,
        _re.IGNORECASE,
    )
    return match is not None and match.group(1).lower() in {
        "grch37",
        "hg19",
        "ncbi37",
    }


async def resolve_gwas_catalog_async(accession, read_text):
    """Resolve an accession using an async text-fetching callback."""

    directory = gwas_catalog_directory(accession)
    root_links = _listing_links(await read_text(directory), directory)
    candidates = []

    harmonised_directory = next(
        (url for url in root_links if url == directory + "harmonised/"), None
    )
    if harmonised_directory:
        links = _listing_links(
            await read_text(harmonised_directory), harmonised_directory
        )
        for suffix in (".h.tsv.gz", ".f.tsv.gz"):
            candidates.extend(
                (url, links)
                for url in sorted(links)
                if _urlsplit(url).path.lower().endswith(suffix)
            )

    candidates.extend(
        (url, root_links)
        for url in sorted(root_links)
        if _re.search(r"\.tsv(?:\.gz)?$", _urlsplit(url).path, _re.IGNORECASE)
    )
    for candidate, links in candidates:
        if _is_build_37_filename(candidate):
            return candidate
        metadata_url = next(
            (url for url in links if url == candidate + "-meta.yaml"), None
        )
        if metadata_url:
            try:
                metadata = await read_text(metadata_url)
            except Exception:
                continue
            if _metadata_is_build_37(metadata):
                return candidate

    raise ValueError(
        "This accession has no GRCh37/hg19 harmonised, formatted, or GWAS-SSF "
        "summary-statistics file compatible with the default LD reference."
    )


def _read_gwas_catalog_text(url):
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    try:
        with urlopen(Request(url, headers={"User-Agent": "VIPRS"})) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding)
    except HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(
                "No GWAS Catalog summary statistics were found for this accession."
            ) from error
        raise ConnectionError(
            f"Could not list GWAS Catalog files (HTTP {error.code})."
        ) from error
    except URLError as error:
        raise ConnectionError(
            "Could not reach the GWAS Catalog FTP server. Check the connection."
        ) from error


def resolve_gwas_catalog(accession, read_text=None):
    """Resolve an accession to its preferred GRCh37 summary-statistics URL.

    ``read_text`` may be supplied to use a different HTTP transport.
    """

    fetch = read_text or _read_gwas_catalog_text

    async def async_fetch(url):
        return fetch(url)

    return _asyncio.run(resolve_gwas_catalog_async(accession, async_fetch))


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


def download_ld_matrix(target_dir=".", chromosome=None):
    """
    Download LD matrices for VIPRS software.

    TODO: Update this once data is made available.

    :param target_dir: The path or directory where to store the LD matrix
    :param chromosome: An integer or list of integers with the chromosome numbers for which to download
    the LD matrices from Zenodo.
    """

    raise NotImplementedError("This function is not yet implemented.")
