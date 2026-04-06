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

## Matriz de versoes fixadas (baseline atual)

Estas versoes foram travadas no script para reduzir risco de incompatibilidade futura:

- Python do host: 3.12.13
- pip: 26.0.1
- setuptools: 70.2.0
- wheel: 0.46.3

- torch: 2.6.0+rocm6.1
- torchvision: 0.21.0+rocm6.1
- torchaudio: 2.6.0+rocm6.1
- pytorch-lightning: 1.9.5
- onnxruntime (pip): 1.24.4
- librosa: 0.11.0
- numpy: 2.4.3
- scipy: 1.17.1
- scikit-learn: 1.8.0
- numba: 0.65.0
- Cython: 0.29.37
- soundfile: 0.13.1
- silero-vad: 6.2.1

- piper_phonemize: commit ba3cc06c5248215928821f1393b2b854a936991a
- piper_train: instalado a partir do repo piper no commit 73c04d81d5590ecc46e522de3601ce7fb29fc2be (subdirectory src/python)

## Importante

- O lock completo de pacotes foi salvo em [requirements.lock.txt](requirements.lock.txt).
- O lock usa referencia remota para piper_train (sem caminho local piper-src).
- Para isso, use o script pronto em [scripts/recriar_env_pipe_tts.sh](scripts/recriar_env_pipe_tts.sh).

## Prerequisitos de sistema (fora do venv)

Os itens abaixo sao necessarios no sistema operacional para o fluxo funcionar:

- python3.12 (interpretador base usado para criar o .venv)
- git (usado para clonar o repositorio piper temporariamente)
- gcc e g++ (necessarios para compilar monotonic_align)
- make (usado no processo de build nativo)
- bibliotecas compartilhadas do sistema:
  - libespeak-ng (usada pelo modulo nativo de fonemizacao)
  - libonnxruntime.so.1.14.1 (usada pelo modulo nativo de fonemizacao)
- stack ROCm/driver AMD instalado no host (necessario para treino com GPU)

Versoes observadas no host atual (referencia):

- git: 2.53.0
- gcc: 15.2.1
- g++: 15.2.1
- make: 4.4.1
- libonnxruntime do sistema: soname 1.14.1

Observacao: esses itens nao sao encapsulados apenas pelo venv e devem existir antes de rodar o script.

## Script completo para recriar

Use exatamente este comando:

```bash
bash scripts/recriar_env_pipe_tts.sh
```

O script faz:

1. Verifica pre-requisitos de sistema (binarios e bibliotecas do host).
2. Cria .venv com Python 3.12.
3. Instala PyTorch ROCm 6.1 fixado.
4. Instala Lightning 1.9.5 e dependencias do pipeline.
5. Instala piper_phonemize por commit.
6. Clona piper em pasta temporaria (tmp), instala piper_train e remove o clone ao final.
7. Compila monotonic_align e copia o .so para site-packages.
8. Aplica patch de compatibilidade no piper_train instalado.
9. Copia silero_vad.onnx do pacote silero-vad para o caminho esperado pelo piper_train.
10. Executa validacoes finais (torch, lightning, piper_train, monotonic_align, silero_vad.onnx).

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
