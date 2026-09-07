"""Deriva os insumos intermediarios que o pipeline espera, a partir dos tres
arquivos publicados em `data/`.

Os scripts de `src/` foram escritos para o layout de trabalho da pesquisa, em que as
features espectrais Sentinel-2 vivem num arquivo proprio, montado pela etapa de
extracao. O pacote publicado ja traz essas colunas mescladas nos dois CSV, entao este
script apenas reverte a mescla e grava o arquivo no caminho esperado.

Alguns scripts tambem procuram a referencia humana e o painel em caminhos fixos, sem
opcao de linha de comando. Para eles este script grava copias dos arquivos publicados
nesses caminhos. Nenhum dado novo e criado, apenas movido para onde o pipeline procura.

Rode uma vez, antes de qualquer experimento:

    python prepare_inputs.py

Saidas:
    data/processed/sentinel2_features.csv
    data/processed/validation/validation_manual_gt.csv
    outputs/ground_truth_validation/painel_labeled_expanded.csv
    outputs/ground_truth_validation/painel_labeled_5strategies.csv
"""

import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
OUT_DIR = DATA_DIR / 'processed'

GT_CSV = DATA_DIR / 'matopiba_fire_human_gt_v1.csv'
PANEL_CSV = DATA_DIR / 'matopiba_fire_full_panel_v1.csv'
POLYGON_GEOJSON = DATA_DIR / 'matopiba_states_polygon.geojson'

# As nove colunas espectrais que `run_decisive_experiment.py` procura para montar o
# conjunto de 22 features. Sem elas o script cai no conjunto de 13 e avisa por
# `[WARN] sentinel2_features.csv nao encontrado`.
SPECTRAL_COLS = [
    'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
    'mean_nbr_pre', 'mean_nbr_post',
    'dnbr_mean', 'dnbr_max', 'pct_burned',
    'land_use_class',
]

KEY_COLS = ['latitude', 'longitude', 'acq_datetime']

# Caminhos fixos que alguns scripts procuram sem oferecer opcao de linha de comando.
# Cada entrada e (arquivo publicado, destino relativo a raiz do repositorio, tirar
# espectrais). Esses scripts leem a referencia humana e trazem as nove colunas
# espectrais do painel por merge, entao a copia da referencia sai sem elas. Manter as
# duas copias faria o merge criar colunas com sufixo e o conjunto de 22 sumiria.
COPIAS = [
    (GT_CSV, Path('data') / 'processed' / 'validation' / 'validation_manual_gt.csv', True),
    (PANEL_CSV, Path('outputs') / 'ground_truth_validation' / 'painel_labeled_5strategies.csv', False),
    (POLYGON_GEOJSON, Path('data') / 'matopiba_limites.geojson', False),
]

# Caminho onde os experimentos de holdout procuram o painel de onde puxam as features
# espectrais por merge. O painel publicado cobre 2022 e a referencia humana vai ate
# dezembro de 2024, entao as 120 deteccoes anotadas de 2023 e 2024 so entram aqui por
# meio da propria referencia. Elas entram sem as nove colunas espectrais, que o pacote
# publicado nao carrega para esses dois anos. A consequencia esta registrada em
# docs/REPRODUCTION.md, na secao de limitacoes.
PAINEL_ESTENDIDO = Path('outputs') / 'ground_truth_validation' / 'painel_labeled_expanded.csv'


def main():
    for caminho in (GT_CSV, PANEL_CSV, POLYGON_GEOJSON):
        if not caminho.exists():
            raise SystemExit(f'ERRO: {caminho} nao existe. Rode a partir da raiz do repositorio.')

    gt = pd.read_csv(GT_CSV)
    panel = pd.read_csv(PANEL_CSV)

    faltando = [c for c in KEY_COLS + SPECTRAL_COLS if c not in panel.columns]
    if faltando:
        raise SystemExit(f'ERRO: {PANEL_CSV.name} nao carrega as colunas {faltando}.')

    keep = KEY_COLS + SPECTRAL_COLS
    espectral = pd.concat([panel[keep], gt[keep]], ignore_index=True)
    espectral = espectral.drop_duplicates(subset=KEY_COLS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destino = OUT_DIR / 'sentinel2_features.csv'
    espectral.to_csv(destino, index=False)

    com_dnbr = int(espectral['dnbr_mean'].notna().sum())
    print(f'[OK] {destino.relative_to(PROJECT_ROOT)}')
    print(f'     {len(espectral)} deteccoes, {com_dnbr} com dNBR calculado')

    alvo_estendido = PROJECT_ROOT / PAINEL_ESTENDIDO
    alvo_estendido.parent.mkdir(parents=True, exist_ok=True)
    comuns = [c for c in panel.columns if c in gt.columns]
    estendido = pd.concat([panel[comuns], gt[comuns]], ignore_index=True)
    estendido = estendido.drop_duplicates(subset=KEY_COLS)
    estendido.to_csv(alvo_estendido, index=False)
    print(f'[OK] {PAINEL_ESTENDIDO}  ({len(estendido)} deteccoes, painel mais as anotadas de 2023 e 2024)')

    for origem, relativo, tirar_espectrais in COPIAS:
        alvo = PROJECT_ROOT / relativo
        alvo.parent.mkdir(parents=True, exist_ok=True)
        if tirar_espectrais:
            base = gt if origem == GT_CSV else pd.read_csv(origem)
            base.drop(columns=SPECTRAL_COLS, errors='ignore').to_csv(alvo, index=False)
            print(f'[OK] {relativo}  (de {origem.name}, sem as colunas espectrais)')
        else:
            shutil.copyfile(origem, alvo)
            print(f'[OK] {relativo}  (copia de {origem.name})')


if __name__ == '__main__':
    main()
