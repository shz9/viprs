"""Generate lightweight browser format guidance from the installed magenpy parsers.

Run with the pinned native magenpy version; no WASM compiler is required.
"""
import json
from pathlib import Path

def get_sumstats_format_help():
    """Describe registered formats using magenpy's own column and output APIs."""
    import pandas as pd
    from magenpy.parsers.sumstats_parsers import SUMSTATS_PARSER_CLASSES

    guides = {
        "magenpy": ("magenpy", "https://shz9.github.io/magenpy/",
                    "A1 is the effect allele. Supply Z, or BETA and SE. MAF contains the frequency of A1."),
        "plink2": ("PLINK 2", "https://www.cog-genomics.org/plink/2.0/formats#glm_logistic",
                   "Keep the # in #CHROM. A1 is the effect allele; the other allele can be inferred from REF and ALT (or ALT1). Keep only the intended additive association rows if TEST contains multiple tests."),
        "plink1.9": ("PLINK 1.9", "https://www.cog-genomics.org/plink/1.9/formats#assoc_linear",
                     "Use tabs for this reader, even if your original PLINK output uses spaces. NMISS is the sample size and STAT is the signed test statistic. The example includes the other allele A2; keep only the intended additive association rows."),
        "cojo": ("GCTA-COJO", "https://yanglab.westlake.edu.cn/software/gcta/#COJO",
                 "Use marginal effects b and se, rather than conditional or joint effects. Add CHR to native COJO files for this browser; the example also includes POS. Use tabs, even if your original file uses spaces."),
        "fastgwa": ("GCTA fastGWA", "https://yanglab.westlake.edu.cn/software/gcta/#fastGWA",
                    "AF1 is the frequency of effect allele A1. N is the per-variant sample size. For binary traits, BETA is the log odds ratio."),
        "ssf": ("GWAS Catalog / GWAS-SSF", "https://www.ebi.ac.uk/gwas/docs/summary-statistics-format/",
                "Use beta and standard_error, and include per-variant sample size n. For odds ratios, provide log(odds ratio) as beta. rsid is optional when chromosome and base_pair_location match unique LD positions. GWAS Catalog accessions select this format automatically."),
        "saige": ("SAIGE", "https://saigegit.github.io/SAIGE-doc/docs/single_step2.html",
                  "Allele2 is the effect allele. This example uses BETA and SE, not the unstandardized score Tstat. Include CHR and POS; N can also be inferred from AC_Allele2 / (2 × AF_Allele2). MarkerID is the recognized SNP-ID header; CHR/POS can identify variants when IDs use another header."),
        "regenie": ("REGENIE", "https://rgcgithub.github.io/regenie/options/#output",
                    "ALLELE1 is the effect allele. LOG10P contains −log10(p), not p itself. Keep only the intended additive association rows if multiple tests are present."),
        "ldak": ("LDAK / SumHer / MegaPRS", "https://dougspeed.com/summary-statistics/",
                 "Supply Z, BETA with SE, Stat with Direction, or P with Direction. Add CHR to native SumHer/MegaPRS files for this browser; the example also includes POS. A1Freq is the frequency of A1."),
    }
    descriptions = {
        "CHR": "Chromosome", "POS": "Base-pair position", "SNP": "Variant ID",
        "A1": "Effect allele", "A2": "Other allele", "MAF": "Effect-allele frequency",
        "N": "Per-variant sample size", "BETA": "Effect size (log odds for binary traits)",
        "Z": "Signed test statistic", "SE": "Standard error of the effect",
        "PVAL": "Association p-value", "CHISQ": "Chi-squared statistic",
        "MAC": "Effect-allele count",
    }
    example = pd.DataFrame({
        "CHR": [22, 22], "POS": [16050075, 16050115], "SNP": ["rs1001", "rs1002"],
        "A1": ["A", "C"], "A2": ["G", "T"], "MAF": [.3, .4], "N": [10000, 10000],
        "BETA": [.1, -.08], "SE": [.02, .02], "Z": [5., -4.],
        "PVAL": [5.733e-7, 6.334e-5], "CHISQ": [25., 16.], "MAC": [6000, 8000],
    })
    help_by_alias = {}
    for parser_type in SUMSTATS_PARSER_CLASSES:
        parser = parser_type(None)
        name, url, notes = guides.get(parser.format_name, (
            parser.format_name, "https://shz9.github.io/magenpy/", "See the format documentation for details."
        ))
        columns = list(dict.fromkeys(("CHR", "POS", *parser.get_standard_cols())))
        if parser.format_name == "saige":
            columns = [column for column in columns if column != "Z"] + ["BETA", "SE"]
        formatted = parser.format_table(example[columns])
        if parser.format_name == "plink2":
            formatted = formatted.rename(columns={"A2": "REF"})
            formatted.insert(4, "ALT", formatted["A1"])
        names = {parser.output_col_name_converter.get(column, column): meaning
                 for column, meaning in descriptions.items()}
        names.update({"REF": "Reference allele", "ALT": "Alternate allele", "LOG10P": "−log10(association p-value)"})
        separator = parser.read_csv_kwargs.get("sep", "\t")
        guide = {
            "name": name, "separator": "Tabs" if separator == "\t" else "Whitespace (spaces or tabs)",
            "separator_symbol": r"\t" if separator == "\t" else r"\s+",
            "columns": [[column, names.get(column, column)] for column in formatted],
            "example": formatted.to_csv(index=False, sep="\t" if separator == "\t" else " ", float_format="%.6g"),
            "notes": notes, "url": url,
        }
        for alias in parser_type.aliases:
            help_by_alias[alias] = guide
    return help_by_alias



def render_format_help():
    return "// Generated by tools/wasm/format_help.py; do not edit by hand.\nexport const SUMSTATS_FORMAT_HELP = " + json.dumps(get_sumstats_format_help(), ensure_ascii=False, separators=(",", ":")) + ";\n"


if __name__ == "__main__":
    (Path(__file__).resolve().parents[2] / "docs/webapp/formats.js").write_text(render_format_help())
