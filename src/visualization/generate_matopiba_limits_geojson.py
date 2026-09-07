"""
Gera o GeoJSON de limites do MATOPIBA usado pelo portal (test_realtime.py).

Estrategia: "estados recortados pela area de estudo".
Usa os contornos REAIS dos 4 estados (Maranhao, Tocantins, Piaui, Bahia)
da base Natural Earth 10m (mesma fonte do mapa da dissertacao) e recorta
cada um pelo bounding box da regiao MATOPIBA. Tocantins fica inteiro
(esta todo dentro do MATOPIBA); MA, PI e BA aparecem apenas na porcao
oeste/sul que compoe a fronteira agricola.

Saida: data/matopiba_limites.geojson (FeatureCollection, 1 feature por
estado, geometria Polygon ou MultiPolygon).

Uso:
    python src/visualization/generate_matopiba_limits_geojson.py
"""

import json
from pathlib import Path

from shapely.geometry import box as shapely_box, mapping
import cartopy.io.shapereader as shpreader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "matopiba_limites.geojson"

# Bounding box da area de estudo MATOPIBA (recorte).
# Mantem TO inteiro + porcoes oeste/sul de MA, PI e BA (fronteira agricola),
# removendo litoral leste/norte que nao faz parte do MATOPIBA.
MATOPIBA_BOUNDS = {
    "west": -50.0,
    "east": -41.5,
    "south": -15.5,
    "north": -1.0,
}

# Tolerancia de simplificacao (graus, ~0.008 deg ~ 900 m) para o mapa web.
SIMPLIFY_TOLERANCE = 0.008

# Estados MATOPIBA: cor + descricao (mesmo padrao do mapa da dissertacao).
MATOPIBA_STATES = {
    "Maranhão": {
        "color": "#D32F2F",
        "description": "Porcao do Maranhao na regiao MATOPIBA",
    },
    "Tocantins": {
        "color": "#1565C0",
        "description": "Tocantins (estado inteiro pertence ao MATOPIBA)",
    },
    "Piauí": {
        "color": "#2E7D32",
        "description": "Porcao do Piaui na regiao MATOPIBA",
    },
    "Bahia": {
        "color": "#EF6C00",
        "description": "Porcao da Bahia na regiao MATOPIBA",
    },
}


def get_brazil_states():
    """Carrega geometrias dos estados brasileiros via Natural Earth 10m."""
    shapefile = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_1_states_provinces"
    )
    reader = shpreader.Reader(shapefile)
    states = {}
    for record in reader.records():
        if record.attributes.get("admin", "") == "Brazil":
            name = record.attributes.get("name", "")
            states[name] = record.geometry
    return states


def main():
    print("[INFO] Carregando estados brasileiros (Natural Earth 10m)...")
    brazil_states = get_brazil_states()

    clip = shapely_box(
        MATOPIBA_BOUNDS["west"],
        MATOPIBA_BOUNDS["south"],
        MATOPIBA_BOUNDS["east"],
        MATOPIBA_BOUNDS["north"],
    )

    features = []
    for name, info in MATOPIBA_STATES.items():
        if name not in brazil_states:
            print(f"  [WARN] estado nao encontrado: {name}")
            continue

        geom = brazil_states[name].intersection(clip)
        if geom.is_empty:
            print(f"  [WARN] {name} nao intersecta a area de estudo")
            continue

        # Simplifica preservando topologia para aliviar o mapa web.
        geom = geom.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "color": info["color"],
                    "description": info["description"],
                },
                "geometry": mapping(geom),
            }
        )
        print(f"  [OK] {name} recortado ({geom.geom_type})")

    fc = {
        "type": "FeatureCollection",
        "properties": {
            "source": "Natural Earth 10m (admin_1_states_provinces), recortado pelo "
            "bounding box MATOPIBA",
            "bounds": MATOPIBA_BOUNDS,
        },
        "features": features,
    }

    OUTPUT_PATH.write_text(
        json.dumps(fc, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] GeoJSON salvo: {OUTPUT_PATH} ({len(features)} estados)")


if __name__ == "__main__":
    main()
