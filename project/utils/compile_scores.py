import json
from pathlib import Path

import pandas as pd


def save_scores(exp_names, exp_results):
    row_header = ['Error', 'mCE',
                  'Gauss', 'Shot', 'Impulse',  # Noise
                  'Defocus', 'Glass', 'Motion', 'Zoom',  # Blur
                  'Snow', 'Frost', 'Fog', 'Bright',  # Weather
                  'Contrast', 'Elastic', 'Pixel', 'JPEG'  # Digital
                  ]
    df1 = pd.DataFrame(index=exp_names, columns=row_header)
    for idx, result in zip(exp_names, exp_results):
        df1['Error'][idx] = 1 - result['mAP']['clean']
        df1['mCE'][idx] = result['mCE']
        df1['Gauss'][idx] = result['CE']['gaussian_noise']
        df1['Shot'][idx] = result['CE']['shot_noise']
        df1['Impulse'][idx] = result['CE']['impulse_noise']
        df1['Defocus'][idx] = result['CE']['defocus_blur']
        df1['Glass'][idx] = result['CE']['glass_blur']
        df1['Motion'][idx] = result['CE']['motion_blur']
        df1['Zoom'][idx] = result['CE']['zoom_blur']
        df1['Snow'][idx] = result['CE']['snow']
        df1['Frost'][idx] = result['CE']['frost']
        df1['Fog'][idx] = result['CE']['fog']
        df1['Bright'][idx] = result['CE']['brightness']
        df1['Contrast'][idx] = result['CE']['contrast']
        df1['Elastic'][idx] = result['CE']['elastic_transform']
        df1['Pixel'][idx] = result['CE']['pixelate']
        df1['JPEG'][idx] = result['CE']['jpeg_compression']

    row_header[1] = 'Rel. mCE'
    df2 = pd.DataFrame(index=exp_names, columns=row_header)
    for idx, result in zip(exp_names, exp_results):
        df2['Error'][idx] = 1 - result['mAP']['clean']
        df2['Rel. mCE'][idx] = result['relative_mCE']
        df2['Gauss'][idx] = result['relative_CE']['gaussian_noise']
        df2['Shot'][idx] = result['relative_CE']['shot_noise']
        df2['Impulse'][idx] = result['relative_CE']['impulse_noise']
        df2['Defocus'][idx] = result['relative_CE']['defocus_blur']
        df2['Glass'][idx] = result['relative_CE']['glass_blur']
        df2['Motion'][idx] = result['relative_CE']['motion_blur']
        df2['Zoom'][idx] = result['relative_CE']['zoom_blur']
        df2['Snow'][idx] = result['relative_CE']['snow']
        df2['Frost'][idx] = result['relative_CE']['frost']
        df2['Fog'][idx] = result['relative_CE']['fog']
        df2['Bright'][idx] = result['relative_CE']['brightness']
        df2['Contrast'][idx] = result['relative_CE']['contrast']
        df2['Elastic'][idx] = result['relative_CE']['elastic_transform']
        df2['Pixel'][idx] = result['relative_CE']['pixelate']
        df2['JPEG'][idx] = result['relative_CE']['jpeg_compression']

    df1.to_csv(r'_csv_results/CE.csv')
    df2.to_csv(r'_csv_results/rel_CE.csv')


def run_flow():
    base_dir = Path(r'/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch_48bit')

    exp_names = [
        'baseline_48-bit-scratch_60epochs',
        '2layers_push3_pull3_avg0_inhibition0.5',
        '2layers_push3_pull3_avg0_inhibition1.0',
        '2layers_push3_pull3_avg0_inhibition1.5',
        '2layers_push3_pull3_avg3_inhibition0.5',
        '2layers_push3_pull3_avg3_inhibition1.0',
        '2layers_push3_pull3_avg3_inhibition1.5',
        '2layers_push3_pull5_avg0_inhibition0.5',
        '2layers_push3_pull5_avg0_inhibition1.0',
        '2layers_push3_pull5_avg0_inhibition1.5',
        '2layers_push3_pull5_avg3_inhibition0.5',
        '2layers_push3_pull5_avg3_inhibition1.0',
        '2layers_push3_pull5_avg3_inhibition1.5',
        '2layers_push5_pull5_avg0_inhibition0.5',
        '2layers_push5_pull5_avg0_inhibition1.0',
        '2layers_push5_pull5_avg0_inhibition1.5',
        '2layers_push5_pull5_avg3_inhibition0.5',
        '2layers_push5_pull5_avg3_inhibition1.0',
        '2layers_push5_pull5_avg3_inhibition1.5',
        '2layers_push5_pull5_avg5_inhibition0.5',
        '2layers_push5_pull5_avg5_inhibition1.0',
        '2layers_push5_pull5_avg5_inhibition1.5',
    ]

    exp_results = []
    for exp_name in exp_names:
        with open(base_dir.joinpath(rf'{exp_name}/results/all_scores.json')) as f:
            exp_results.append(json.load(f))

    save_scores(exp_names, exp_results)


if __name__ == '__main__':
    run_flow()
