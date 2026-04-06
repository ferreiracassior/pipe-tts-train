# Recriar Ambiente Python do Pipe TTS Train

Este documento descreve como recriar o ambiente virtual do projeto com o estado funcional atual (treino + preprocess + export), sem manter piper-src dentro do repositorio.

## Resumo do ambiente atual

- Sistema: Linux
- Projeto: /home/cassio/git/pipe-tts-train
- Python: 3.12.13
- Ambiente: venv em .venv
- GPU detectada: AMD Radeon RX 7800 XT
- Stack principal:
  - torch 2.6.0+rocm6.1
  - torchvision 0.21.0+rocm6.1
  - torchaudio 2.6.0+rocm6.1
  - pytorch-lightning 1.9.5
  - piper_train 1.0.0
  - piper_phonemize (commit fixado)
  - silero-vad 6.2.1
  - onnxruntime 1.24.4

## Importante

- O lock completo de pacotes foi salvo em [requirements.lock.txt](requirements.lock.txt).
- O lock usa referencia remota para piper_train (sem caminho local piper-src).
- Para isso, use o script pronto em [scripts/recriar_env_pipe_tts.sh](scripts/recriar_env_pipe_tts.sh).

## Script completo para recriar

Use exatamente este comando:

```bash
bash scripts/recriar_env_pipe_tts.sh
```

O script faz:

1. Cria .venv com Python 3.12.
2. Instala PyTorch ROCm 6.1 fixado.
3. Instala Lightning 1.9.5 e dependencias do pipeline.
4. Instala piper_phonemize por commit.
5. Clona piper em pasta temporaria (tmp), instala piper_train e remove o clone ao final.
6. Compila monotonic_align e copia o .so para site-packages.
7. Aplica patch de compatibilidade no piper_train instalado.
8. Copia silero_vad.onnx do pacote silero-vad para o caminho esperado pelo piper_train.
9. Executa validacoes finais (torch, lightning, piper_train, monotonic_align, silero_vad.onnx).

## Execucao de treino apos recriar

```bash
/home/cassio/git/pipe-tts-train/.venv/bin/python treinar_vozes_piper.py \
  --max-epochs-cap 2000 \
  --max-train-hours 2 \
  --batch-size 16 \
  --preprocess-max-workers 16
```

## Checklist de validacao manual

Depois de recriar, rode:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
.venv/bin/python -c "import pytorch_lightning as pl; print(pl.__version__)"
.venv/bin/python -c "import piper_train; print(piper_train.__file__)"
.venv/bin/python -c "import piper_train.vits.monotonic_align as m; print(m.__file__)"
.venv/bin/python -c "from pathlib import Path; import piper_train; p=Path(piper_train.__file__).resolve().parent/'norm_audio'/'models'/'silero_vad.onnx'; print(p.exists(), p)"
```

Se todas responderem sem erro e com `True` no silero_vad, o ambiente esta pronto.

## Observacoes sobre reproducibilidade

- O piper_train oficial nao esta publicado de forma simples no PyPI para este fluxo, por isso o script usa clone temporario.
- O repositorio principal permanece limpo (sem piper-src persistente).
- O arquivo de lock [requirements.lock.txt](requirements.lock.txt) serve para auditoria do estado atual.
