import os
import subprocess
import shutil
from datetime import datetime

import pandas as pd

# Configurações de Caminho
CV_PT_DIR = os.path.join('datasets', 'cv-corpus-25.0-2026-03-09', 'pt')
TSV_FILE = os.path.join(CV_PT_DIR, 'validated.tsv')
INPUT_DIR = os.path.join(CV_PT_DIR, 'clips')
BASE_OUTPUT_DIR = 'piper_dataset_por_voz'


def pct(atual, total):
    if total == 0:
        return 0.0
    return (atual / total) * 100


def barra_progresso(atual, total, largura=32):
    if total == 0:
        preenchido = 0
    else:
        preenchido = int((atual / total) * largura)
    vazios = max(0, largura - preenchido)
    return f"[{'#' * preenchido}{'.' * vazios}]"


def imprimir_dashboard(stats):
    # Limpa a tela para atualizar o dashboard em tempo real.
    print("\033[2J\033[H", end="")
    largura = max(80, shutil.get_terminal_size((100, 30)).columns)
    separador = "=" * largura

    vozes_pct = pct(stats['vozes_processadas'], stats['total_vozes'])
    clips_pct = pct(stats['clips_processados'], stats['total_clips'])
    voz_atual_pct = pct(stats['voz_clips_processados'], stats['voz_clips_total'])

    print(separador)
    print(" DASHBOARD - PREPARO DE DATASET TTS (PT-BR) ".center(largura))
    print(separador)
    print(f"Iniciado em........: {stats['inicio']}")
    print(f"Arquivo TSV........: {TSV_FILE}")
    print(f"Pasta de saída.....: {BASE_OUTPUT_DIR}")
    print("-" * largura)
    print(f"Vozes processadas..: {stats['vozes_processadas']}/{stats['total_vozes']} ({vozes_pct:6.2f}%) {barra_progresso(stats['vozes_processadas'], stats['total_vozes'])}")
    print(f"Clips analisados...: {stats['clips_processados']}/{stats['total_clips']} ({clips_pct:6.2f}%) {barra_progresso(stats['clips_processados'], stats['total_clips'])}")
    print("-" * largura)
    print(f"Convertidos........: {stats['clips_convertidos']}")
    print(f"Origem ausente.....: {stats['clips_sem_origem']}")
    print(f"Falhas no ffmpeg...: {stats['clips_falha_ffmpeg']}")
    print("-" * largura)
    print(f"Voz atual..........: {stats['voz_atual']}")
    print(f"Progresso da voz...: {stats['voz_clips_processados']}/{stats['voz_clips_total']} ({voz_atual_pct:6.2f}%) {barra_progresso(stats['voz_clips_processados'], stats['voz_clips_total'])}")
    print(separador)


# Carregar e Filtrar
df = pd.read_csv(TSV_FILE, sep='\t', low_memory=False)
df = df[df['accents'].str.contains("Brasil", na=False)]

# Pegar todos os client_ids únicos
todos_clientes = df['client_id'].unique()

stats = {
    'inicio': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
    'total_vozes': len(todos_clientes),
    'vozes_processadas': 0,
    'total_clips': len(df),
    'clips_processados': 0,
    'clips_convertidos': 0,
    'clips_sem_origem': 0,
    'clips_falha_ffmpeg': 0,
    'voz_atual': '-',
    'voz_clips_total': 0,
    'voz_clips_processados': 0,
}

imprimir_dashboard(stats)

for v_id in todos_clientes:
    # ID Curto para a pasta (8 chars)
    short_id = v_id[:8]
    voice_dir = os.path.join(BASE_OUTPUT_DIR, f"voice_{short_id}")
    wav_dir = os.path.join(voice_dir, 'wavs')

    voice_df = df[df['client_id'] == v_id]
    metadata = []
    clips_total_voz = len(voice_df)

    # Só cria a pasta se houver áudios válidos
    pasta_criada = False

    stats['voz_atual'] = short_id
    stats['voz_clips_total'] = clips_total_voz
    stats['voz_clips_processados'] = 0
    imprimir_dashboard(stats)

    for _, row in voice_df.iterrows():
        mp3_path = os.path.join(INPUT_DIR, row['path'])
        wav_filename = row['path'].replace('.mp3', '.wav')
        wav_path = os.path.join(wav_dir, wav_filename)

        if os.path.exists(mp3_path):
            if not pasta_criada:
                os.makedirs(wav_dir, exist_ok=True)
                pasta_criada = True

            # Conversão FFmpeg (22050Hz, Mono)
            resultado = subprocess.run([
                'ffmpeg', '-y', '-i', mp3_path,
                '-ar', '22050', '-ac', '1',
                wav_path, '-loglevel', 'error'
            ], check=False)

            if resultado.returncode == 0:
                stats['clips_convertidos'] += 1
                metadata.append(f"{wav_filename.replace('.wav', '')}|{row['sentence'].strip()}")
            else:
                stats['clips_falha_ffmpeg'] += 1
        else:
            stats['clips_sem_origem'] += 1

        stats['clips_processados'] += 1
        stats['voz_clips_processados'] += 1

        # Reduz flicker: atualiza com menos frequência em vozes grandes.
        if (
            stats['voz_clips_processados'] == stats['voz_clips_total']
            or stats['voz_clips_processados'] % 25 == 0
        ):
            imprimir_dashboard(stats)

    if pasta_criada and metadata:
        with open(os.path.join(voice_dir, 'metadata.csv'), 'w', encoding='utf-8') as f:
            for line in metadata:
                f.write(line + '\n')

    stats['vozes_processadas'] += 1
    imprimir_dashboard(stats)

print("\nConcluído! Todos os datasets individuais foram gerados.")