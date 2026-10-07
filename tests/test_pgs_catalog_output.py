import gzip

import pandas as pd
import pytest

from viprs.model.BayesPRSModel import BayesPRSModel


@pytest.fixture
def fitted_model():
    model = BayesPRSModel.__new__(BayesPRSModel)
    parameter_table = pd.DataFrame({
        'CHR': [1, 2],
        'SNP': ['rs1', 'rs2'],
        'POS': [101, 202],
        'A1': ['A', 'C'],
        'A2': ['G', 'T'],
        'BETA': [0.25, -0.5],
    })
    model.to_table = lambda per_chromosome=False: parameter_table.copy()
    return model


def test_write_pgs_catalog_scoring_file(fitted_model, tmp_path):
    output_prefix = tmp_path / 'score'
    fitted_model.write_inferred_parameters(
        output_prefix,
        output_format='pgs_catalog',
        pgs_metadata={
            'genome_build': 'GRCh37',
            'pgs_name': 'VIPRS_height',
            'trait_reported': 'Height',
        },
    )

    output_file = tmp_path / 'score.txt.gz'
    with gzip.open(output_file, 'rt', encoding='utf-8') as score_file:
        contents = score_file.read()

    assert '#format_version=2.0\n' in contents
    assert '#pgs_id=PGS000000\n' in contents
    assert '#pgs_name=VIPRS_height\n' in contents
    assert '#genome_build=GRCh37\n' in contents
    assert '#variants_number=2\n' in contents

    table = pd.read_csv(output_file, sep='\t', comment='#')
    assert list(table.columns) == [
        'rsID', 'chr_name', 'chr_position', 'effect_allele', 'other_allele',
        'effect_weight',
    ]
    assert table.to_dict(orient='records') == [
        {
            'rsID': 'rs1', 'chr_name': 1, 'chr_position': 101,
            'effect_allele': 'A', 'other_allele': 'G', 'effect_weight': 0.25,
        },
        {
            'rsID': 'rs2', 'chr_name': 2, 'chr_position': 202,
            'effect_allele': 'C', 'other_allele': 'T', 'effect_weight': -0.5,
        },
    ]


def test_pgs_catalog_output_requires_genome_build(fitted_model, tmp_path):
    with pytest.raises(ValueError, match='genome_build'):
        fitted_model.write_inferred_parameters(
            tmp_path / 'score',
            output_format='pgs_catalog',
        )


def test_pgs_catalog_output_can_select_model(fitted_model, tmp_path):
    original_to_table = fitted_model.to_table
    fitted_model.to_table = lambda per_chromosome=False: original_to_table().assign(
        BETA_1=[1.5, 2.5]
    )

    output_file = tmp_path / 'selected.txt'
    fitted_model.write_inferred_parameters(
        output_file,
        output_format='pgs_catalog',
        pgs_metadata={'genome_build': 'GRCh38'},
        effect_column='BETA_1',
    )

    table = pd.read_csv(output_file, sep='\t', comment='#')
    assert table['effect_weight'].tolist() == [1.5, 2.5]


def test_native_output_remains_the_default(fitted_model, tmp_path):
    output_prefix = str(tmp_path / 'score')
    fitted_model.write_inferred_parameters(output_prefix)

    table = pd.read_csv(f'{output_prefix}.fit', sep='\t')
    assert list(table.columns) == ['CHR', 'SNP', 'POS', 'A1', 'A2', 'BETA']


def test_read_pgs_catalog_scoring_file(fitted_model, tmp_path):
    output_file = tmp_path / 'score.txt.gz'
    fitted_model.write_inferred_parameters(
        output_file,
        output_format='pgs_catalog',
        pgs_metadata={'genome_build': 'GRCh37'},
    )

    loaded_tables = []
    model = BayesPRSModel.__new__(BayesPRSModel)
    model.set_model_parameters = loaded_tables.append
    model.read_inferred_parameters(output_file, input_format='pgs_catalog')

    table = loaded_tables[0]
    assert list(table.columns) == ['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA']
    assert table['BETA'].tolist() == [0.25, -0.5]


def test_read_native_parameters_remains_the_default(fitted_model, tmp_path):
    output_prefix = str(tmp_path / 'score')
    fitted_model.write_inferred_parameters(output_prefix)

    loaded_tables = []
    model = BayesPRSModel.__new__(BayesPRSModel)
    model.set_model_parameters = loaded_tables.append
    model.read_inferred_parameters(f'{output_prefix}.fit')

    assert loaded_tables[0]['BETA'].tolist() == [0.25, -0.5]


def test_read_parameters_rejects_unknown_format():
    model = BayesPRSModel.__new__(BayesPRSModel)
    with pytest.raises(ValueError, match='input_format'):
        model.read_inferred_parameters('score.txt', input_format='unknown')
