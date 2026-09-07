"""
Generate improved GEE validation script with navigation panel.
Replaces the original click-based script with a proper UI navigator.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

VALIDATION_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed' / 'validation'

def generate_gee_navigator_script():
    csv_path = VALIDATION_DIR / 'validation_sample.csv'
    df = pd.read_csv(csv_path)

    # Build client-side data array from CSV
    data_lines = []
    for _, row in df.iterrows():
        sid = int(row['sample_id'])
        lon = row['longitude']
        lat = row['latitude']
        wl = int(row['is_reliable'])  # 1=real (reliable), 0=spurious (unreliable)
        date_str = str(row['acq_datetime'])[:10]
        data_lines.append(
            f"  {{lon:{lon},lat:{lat},id:'{sid}',wl:{wl},date:'{date_str}'}}"
        )

    data_array = ',\n'.join(data_lines)
    total = len(df)

    script = f"""// ===== VALIDACAO WEAK LABELS - SCRIPT COM NAVEGACAO =====
// Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}
// Total de amostras: {total}
//
// COMO USAR:
// 1. Cole este script no Google Earth Engine Code Editor
// 2. Clique "Run" para executar
// 3. Use o PAINEL A DIREITA para navegar entre pontos
// 4. Clique "Proximo >" para avancar ou "< Anterior" para voltar
// 5. Para cada ponto, analise a camada dNBR:
//    - Vermelho/Laranja = queimada CONFIRMADA -> manual_label = 0
//    - Verde = SEM queimada -> manual_label = 1
//    - Amarelo = incerto, analise o RGB pos-fogo
// 6. Anote sua classificacao no CSV (validation_sample.csv)

// ========== DADOS DAS AMOSTRAS (client-side para navegacao rapida) ==========
var DATA = [
{data_array}
];

var totalSamples = DATA.length;
var currentIndex = 0;

// ========== CONFIGURACOES DE VISUALIZACAO ==========
var rgbVis = {{bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}};
var dnbrVis = {{min: -0.3, max: 0.6, palette: ['green', 'yellow', 'orange', 'red']}};

// ========== FUNCOES DE IMAGEM ==========
function loadImagery(lon, lat, dateStr) {{
  var point = ee.Geometry.Point([lon, lat]);
  var geometry = point.buffer(2000);
  var date = ee.Date(dateStr);

  var preFire = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geometry)
    .filterDate(date.advance(-60, 'day'), date.advance(-30, 'day'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .median()
    .clip(geometry);

  var postFire = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geometry)
    .filterDate(date, date.advance(30, 'day'))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .median()
    .clip(geometry);

  var preNBR = preFire.normalizedDifference(['B8', 'B12']).rename('NBR');
  var postNBR = postFire.normalizedDifference(['B8', 'B12']).rename('NBR');
  var dNBR = preNBR.subtract(postNBR).rename('dNBR');

  return {{pre: preFire, post: postFire, dnbr: dNBR, point: point}};
}}

// ========== PAINEL DE NAVEGACAO ==========
var panel = ui.Panel({{
  style: {{width: '330px', padding: '8px'}}
}});

var titleLabel = ui.Label('VALIDACAO WEAK LABELS', {{
  fontWeight: 'bold', fontSize: '16px', margin: '0 0 4px 0', color: '#1a73e8'
}});
var subtitleLabel = ui.Label('Navegue entre os ' + totalSamples + ' pontos', {{
  fontSize: '12px', color: '#666', margin: '0 0 10px 0'
}});

var progressLabel = ui.Label('Ponto 1 / ' + totalSamples, {{
  fontWeight: 'bold', fontSize: '14px'
}});
var progressBar = ui.Label('', {{
  backgroundColor: '#1a73e8', height: '4px',
  width: (1/totalSamples*100) + '%', margin: '2px 0 8px 0'
}});

// Info labels
var idLabel = ui.Label('');
var weakLabelText = ui.Label('');
var dateLabel = ui.Label('');
var coordsLabel = ui.Label('');
var statusLabel = ui.Label('Carregando...', {{color: 'orange', fontWeight: 'bold'}});

// Separator
var sep1 = ui.Label('________________________________________', {{
  color: '#ccc', fontSize: '10px', margin: '4px 0'
}});
var sep2 = ui.Label('________________________________________', {{
  color: '#ccc', fontSize: '10px', margin: '4px 0'
}});

// Navigation buttons
var prevBtn = ui.Button({{
  label: '< Anterior',
  onClick: function() {{ goToPoint(currentIndex - 1); }},
  style: {{stretch: 'horizontal'}}
}});
var nextBtn = ui.Button({{
  label: 'Proximo >',
  onClick: function() {{ goToPoint(currentIndex + 1); }},
  style: {{stretch: 'horizontal', color: 'white', backgroundColor: '#1a73e8'}}
}});
var navRow = ui.Panel([prevBtn, nextBtn], ui.Panel.Layout.flow('horizontal'), {{
  margin: '4px 0'
}});

// Jump to point
var jumpInput = ui.Textbox({{
  placeholder: 'Numero (1-' + totalSamples + ')',
  style: {{stretch: 'horizontal'}}
}});
var jumpBtn = ui.Button({{
  label: 'Ir',
  onClick: function() {{
    var val = parseInt(jumpInput.getValue(), 10);
    if (!isNaN(val) && val >= 1 && val <= totalSamples) {{
      goToPoint(val - 1);
    }}
  }}
}});
var jumpRow = ui.Panel([jumpInput, jumpBtn], ui.Panel.Layout.flow('horizontal'), {{
  margin: '0 0 4px 0'
}});

// Interpretation guide
var guideTitle = ui.Label('GUIA RAPIDO:', {{
  fontWeight: 'bold', fontSize: '12px', margin: '4px 0 2px 0'
}});
var guide1 = ui.Label('dNBR Vermelho/Laranja = QUEIMADA (label=0)', {{
  fontSize: '11px', color: '#d32f2f'
}});
var guide2 = ui.Label('dNBR Verde = SEM queimada (label=1)', {{
  fontSize: '11px', color: '#388e3c'
}});
var guide3 = ui.Label('dNBR Amarelo = Incerto (analisar RGB)', {{
  fontSize: '11px', color: '#f57f17'
}});
var guideTip = ui.Label('Ative "Post-fire RGB" nas camadas para ver a imagem real.', {{
  fontSize: '10px', color: '#999', margin: '4px 0 0 0'
}});

// Assemble panel
panel.add(titleLabel);
panel.add(subtitleLabel);
panel.add(progressLabel);
panel.add(progressBar);
panel.add(navRow);
panel.add(jumpRow);
panel.add(sep1);
panel.add(idLabel);
panel.add(weakLabelText);
panel.add(dateLabel);
panel.add(coordsLabel);
panel.add(statusLabel);
panel.add(sep2);
panel.add(guideTitle);
panel.add(guide1);
panel.add(guide2);
panel.add(guide3);
panel.add(guideTip);

ui.root.add(panel);

// ========== FUNCAO PRINCIPAL DE NAVEGACAO ==========
function goToPoint(index) {{
  if (index < 0 || index >= totalSamples) return;
  currentIndex = index;
  var d = DATA[index];

  // Update info panel
  progressLabel.setValue('Ponto ' + (index + 1) + ' / ' + totalSamples);
  progressBar.style().set('width', ((index + 1) / totalSamples * 100) + '%');
  idLabel.setValue('Sample ID: ' + d.id);
  weakLabelText.setValue('Weak Label: ' + (d.wl === 1 ? '1 - Real (confiavel)' : '0 - Espurio (nao confiavel)'));
  dateLabel.setValue('Data do foco: ' + d.date);
  coordsLabel.setValue('Lat: ' + d.lat.toFixed(5) + '  Lon: ' + d.lon.toFixed(5));
  statusLabel.setValue('Carregando imagens...');
  statusLabel.style().set('color', 'orange');

  // Clear all layers
  Map.layers().reset();

  // Center map on point
  var point = ee.Geometry.Point([d.lon, d.lat]);
  Map.centerObject(point, 14);

  // Add marker for current point (yellow)
  Map.addLayer(point, {{color: 'FFFF00'}}, 'Ponto #' + (index+1) + ' (ID: ' + d.id + ')');

  // Load and display imagery
  var imagery = loadImagery(d.lon, d.lat, d.date);

  // Add layers (dNBR visible by default, RGB hidden)
  Map.addLayer(imagery.pre, rgbVis, 'Pre-fire RGB', false);
  Map.addLayer(imagery.post, rgbVis, 'Post-fire RGB', false);
  Map.addLayer(imagery.dnbr, dnbrVis, 'dNBR (queimada)', true);

  // Update status
  statusLabel.setValue('Pronto! Analise o dNBR.');
  statusLabel.style().set('color', 'green');

  // Print to console for tracking
  print('--- Ponto ' + (index+1) + '/' + totalSamples + ' ---');
  print('ID: ' + d.id + ' | Weak Label: ' + d.wl + ' | Data: ' + d.date);
}}

// ========== INICIAR ==========
Map.setOptions('SATELLITE');
goToPoint(0);
print('=== VALIDACAO WEAK LABELS ===');
print('Use o painel a direita para navegar.');
print('Total: ' + totalSamples + ' amostras.');
"""

    output_path = VALIDATION_DIR / 'validation_gee_navigator.js'
    output_path.write_text(script, encoding='utf-8')
    print(f"Script gerado: {output_path}")
    print(f"Total de amostras: {total}")
    print(f"Tamanho do arquivo: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


if __name__ == '__main__':
    generate_gee_navigator_script()
