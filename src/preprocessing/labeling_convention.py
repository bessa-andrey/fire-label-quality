"""Convencao de rotulagem: projecao da referencia humana sobre o eixo do construto.

CONTEXTO
--------
A dissertacao adota a Convencao *sensu sensor*, formalizada na Secao de definicoes
operacionais e detalhada no Capitulo 3 da dissertacao, na secao Convencao de
Rotulagem: houve combustao de biomassa no pixel, entao
`y = 0` (fogo real), independentemente da intencionalidade, do uso do solo ou da
relevancia operacional. Queimada agricola controlada, queimada prescrita e manejo de
pastagem sao, portanto, fogo real.

A alternativa declarada no mesmo trecho e a Convencao *sensu gestor*: so conta como
relevante o fogo nao intencional em vegetacao, entao a queimada de manejo migra para a
classe espuria. Este modulo e a fonte unica de verdade dessa projecao, no mesmo espirito
de `src/preprocessing/dnbr_rule.py`. Nao reimplemente a regra em scripts de analise.

O QUE A CONVENCAO ALCANCA, E O QUE NAO ALCANCA
----------------------------------------------
A convencao redefine o CONSTRUTO, e o unico rotulo que pode ser reprojetado sobre o
construto novo e o da referencia humana, porque so ele carrega o registro do contexto do
fogo. As fontes automaticas (weak labels, dNBR automatico, Painel do Fogo, MapBiomas
Fogo) nao distinguem intencao: elas medem cicatriz, e continuam medindo exatamente a
mesma coisa sob as duas convencoes. Segue disso que os pares entre fontes automaticas
NAO mudam de valor, e a analise de sensibilidade se concentra nos pares que envolvem a
referencia humana. Reprojetar tambem as fontes automaticas pela mascara de notas seria
injetar informacao do anotador dentro delas.

AS DUAS MASCARAS
----------------
`notas`     Mascara declarada no texto do Cap. 3. Usa a taxonomia registrada em
            `human_notes` durante a anotacao cega: contexto agricola, queimada
            prescrita ou manejo. Alcanca 108 das 535 amostras binarias, todas elas
            rotuladas fogo real. E um PISO, nao um censo: o campo de nota livre foi
            preenchido em 144 das 535, logo o contingente de manejo efetivo e maior.

`uso_solo`  Mascara por classe de uso do solo do MapBiomas (`land_use_class`), que
            existe no painel e nao depende do anotador. Alcanca uma fracao muito maior
            da amostra, porem troca "o anotador viu manejo" por "o pixel esta em classe
            antropica", que e uma proxy mais larga: fogo acidental dentro de pastagem
            tambem cai nela. Fica disponivel como analise de robustez, e NAO e a
            operacionalizacao declarada no texto.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

CONVENCOES = ('sensu_sensor', 'sensu_gestor')
MASCARAS = ('notas', 'uso_solo')

# Termos da taxonomia de notas do protocolo de anotacao que indicam combustao de manejo.
# Ver a enumeracao da taxonomia no Cap. 3: "Queimada agricola controlada",
# "Queimada prescrita / manejo do fogo" e "Fogo irregular em area agricola".
TERMOS_MANEJO = ('agricol', 'prescrita', 'manejo')

# Classes antropicas do MapBiomas colecao 8 usadas pela mascara `uso_solo`.
# 9 silvicultura, 15 pastagem, 20 cana, 21 mosaico de agricultura e pastagem,
# 35 dende, 39 soja, 40 arroz, 41 outras lavouras temporarias, 46 cafe,
# 48 outras lavouras perenes.
CLASSES_ANTROPICAS = (9, 15, 20, 21, 35, 39, 40, 41, 46, 48)


def _sem_acento(valor) -> str:
    texto = '' if valor is None or (isinstance(valor, float) and np.isnan(valor)) else str(valor)
    return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode().lower().strip()


def mascara_manejo(df: pd.DataFrame, mascara: str = 'notas',
                   coluna_notas: str = 'human_notes',
                   coluna_uso_solo: str = 'land_use_class') -> pd.Series:
    """Marca as linhas em que a combustao observada e de manejo.

    Retorna uma Series booleana alinhada ao indice de `df`.
    """
    if mascara not in MASCARAS:
        raise ValueError(f'mascara desconhecida: {mascara}. Use uma de {MASCARAS}')

    if mascara == 'notas':
        if coluna_notas not in df.columns:
            raise KeyError(
                f'a mascara `notas` exige a coluna {coluna_notas} na referencia humana')
        notas = df[coluna_notas].map(_sem_acento)
        marca = pd.Series(False, index=df.index)
        for termo in TERMOS_MANEJO:
            marca = marca | notas.str.contains(termo, regex=False)
        return marca

    if coluna_uso_solo not in df.columns:
        raise KeyError(
            f'a mascara `uso_solo` exige a coluna {coluna_uso_solo}, que vem do painel')
    return df[coluna_uso_solo].isin(CLASSES_ANTROPICAS)


def aplicar_convencao(df: pd.DataFrame, convencao: str = 'sensu_sensor',
                      coluna_rotulo: str = 'human_label',
                      mascara: str = 'notas', **kwargs):
    """Projeta a referencia humana sobre o eixo da convencao pedida.

    Sob `sensu_sensor` devolve o rotulo intacto, que e o comportamento de toda a
    dissertacao. Sob `sensu_gestor` reclassifica como espuria (`1`) toda amostra em que
    houve combustao de manejo. Amostras incertas (`-1`) nao sao tocadas, seguem excluidas
    do calculo de concordancia como manda o protocolo.

    Retorna `(rotulos, marca)`, com `marca` booleana das linhas efetivamente movidas.
    """
    if convencao not in CONVENCOES:
        raise ValueError(f'convencao desconhecida: {convencao}. Use uma de {CONVENCOES}')

    rotulos = df[coluna_rotulo]
    if convencao == 'sensu_sensor':
        return rotulos.copy(), pd.Series(False, index=df.index)

    manejo = mascara_manejo(df, mascara=mascara, **kwargs)
    movidas = manejo & rotulos.isin([0, 1]) & (rotulos != 1)
    return rotulos.where(~movidas, 1), movidas
