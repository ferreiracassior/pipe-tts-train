#!/usr/bin/env bash
set -euo pipefail

# Recria o ambiente Python do projeto pipe-tts-train
# sem manter piper-src dentro do workspace.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="python3.12"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERRO: python3.12 nao encontrado no sistema."
  exit 1
fi

echo "[1/11] Criando venv em $VENV_DIR"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

echo "[2/11] Atualizando ferramentas base"
"$PIP" install --upgrade pip setuptools wheel

echo "[3/11] Instalando stack PyTorch ROCm"
"$PIP" install --index-url https://download.pytorch.org/whl/rocm6.1 \
  torch==2.6.0+rocm6.1 torchvision==0.21.0+rocm6.1 torchaudio==2.6.0+rocm6.1

echo "[4/11] Instalando dependencias gerais"
"$PIP" install \
  pytorch-lightning==1.9.5 \
  onnxruntime==1.24.4 \
  librosa==0.11.0 \
  numpy==2.4.3 \
  scipy==1.17.1 \
  scikit-learn==1.8.0 \
  numba==0.65.0 \
  Cython==0.29.37 \
  soundfile==0.13.1 \
  silero-vad==6.2.1

echo "[5/11] Instalando piper_phonemize (fixado)"
"$PIP" install "piper_phonemize @ git+https://github.com/rhasspy/piper-phonemize.git@ba3cc06c5248215928821f1393b2b854a936991a"

echo "[6/11] Baixando piper temporariamente para instalar piper_train"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone https://github.com/rhasspy/piper.git "$TMP_DIR/piper"

# Instala piper_train no venv sem deps (deps ja foram controladas acima)
"$PIP" install --no-deps "$TMP_DIR/piper/src/python"

echo "[7/11] Compilando monotonic_align"
(
  cd "$TMP_DIR/piper/src/python"
  "$PY" piper_train/vits/monotonic_align/setup.py build_ext --inplace
)

SO_FILE="$(ls "$TMP_DIR/piper/src/python/piper_train/vits/monotonic_align"/core.cpython-*.so | head -n 1)"
if [ -z "$SO_FILE" ]; then
  echo "ERRO: arquivo .so de monotonic_align nao foi gerado."
  exit 1
fi

SITE_PKGS="$($PY -c 'import site; print(site.getsitepackages()[0])')"
PIPER_TRAIN_DIR="$SITE_PKGS/piper_train"
MONO_DIR="$PIPER_TRAIN_DIR/vits/monotonic_align"
mkdir -p "$MONO_DIR"
cp -f "$SO_FILE" "$MONO_DIR/"

echo "[8/11] Aplicando patch de compatibilidade no piper_train"
"$PY" - <<PYCODE
from pathlib import Path

site_pkgs = Path(r"$SITE_PKGS")
main_py = site_pkgs / "piper_train" / "__main__.py"
mono_init = site_pkgs / "piper_train" / "vits" / "monotonic_align" / "__init__.py"

src = main_py.read_text(encoding="utf-8")

if "Trainer.add_argparse_args(parser)" in src:
    src = src.replace(
        "    Trainer.add_argparse_args(parser)\n"
        "    VitsModel.add_model_specific_args(parser)\n"
        "    parser.add_argument(\"--seed\", type=int, default=1234)\n"
        "    args = parser.parse_args()\n",
        "    # Trainer.add_argparse_args/from_argparse_args were removed in Lightning 2.x.\n"
        "    parser.add_argument(\"--accelerator\", default=\"cpu\", help=\"Accelerator to use (cpu, gpu, etc)\")\n"
        "    parser.add_argument(\"--devices\", type=int, default=1, help=\"Number of devices\")\n"
        "    parser.add_argument(\"--gpus\", type=int, default=None, help=\"Number of GPUs (deprecated, use devices)\")\n"
        "    parser.add_argument(\"--max_epochs\", type=int, default=1000, help=\"Maximum number of epochs\")\n"
        "    parser.add_argument(\"--precision\", default=\"32-true\", help=\"Precision (32-true, 16-mixed, etc)\")\n"
        "    parser.add_argument(\"--batch_size\", type=int, default=32, help=\"Batch size\")\n"
        "    parser.add_argument(\"--num_sanity_val_steps\", type=int, default=2, help=\"Number of sanity validation steps\")\n"
        "    parser.add_argument(\"--limit_val_batches\", type=int, default=1.0, help=\"Limit validation batches\")\n"
        "    parser.add_argument(\"--validation_split\", type=float, default=0.1, help=\"Validation split\")\n"
        "    parser.add_argument(\"--num_test_examples\", type=int, default=0, help=\"Number of test examples\")\n"
        "    parser.add_argument(\"--default_root_dir\", default=None, help=\"Default root directory\")\n"
        "    parser.add_argument(\"--enable_checkpointing\", type=bool, default=True, help=\"Enable checkpointing\")\n"
        "    parser.add_argument(\"--logger\", type=bool, default=True, help=\"Enable logger\")\n"
        "    parser.add_argument(\"--gradient_clip_val\", type=float, default=None, help=\"Gradient clip value\")\n"
        "    VitsModel.add_model_specific_args(parser)\n"
        "    parser.add_argument(\"--seed\", type=int, default=1234)\n"
        "    parser.set_defaults(\n"
        "        num_sanity_val_steps=0,\n"
        "        limit_val_batches=0,\n"
        "        validation_split=0,\n"
        "        num_test_examples=0,\n"
        "    )\n"
        "    args = parser.parse_args()\n"
        "    args.num_sanity_val_steps = 0\n"
        "    args.limit_val_batches = 0\n"
        "    args.validation_split = 0\n"
        "    args.num_test_examples = 0\n"
    )

if "trainer = Trainer.from_argparse_args(args)" in src:
    src = src.replace(
        "    trainer = Trainer.from_argparse_args(args)\n",
        "    trainer_kwargs = {\n"
        "        \"accelerator\": args.accelerator,\n"
        "        \"devices\": args.devices,\n"
        "        \"max_epochs\": args.max_epochs,\n"
        "        \"precision\": args.precision,\n"
        "        \"default_root_dir\": args.default_root_dir or args.dataset_dir,\n"
        "        \"enable_checkpointing\": args.enable_checkpointing,\n"
        "        \"logger\": args.logger,\n"
        "        \"gradient_clip_val\": args.gradient_clip_val,\n"
        "        \"num_sanity_val_steps\": args.num_sanity_val_steps,\n"
        "    }\n"
        "    trainer = Trainer(**trainer_kwargs)\n"
    )

main_py.write_text(src, encoding="utf-8")

mono_src = mono_init.read_text(encoding="utf-8")
if "from .monotonic_align.core import maximum_path_c" in mono_src:
    mono_src = mono_src.replace(
        "from .monotonic_align.core import maximum_path_c\n",
        "try:\n"
        "    from .monotonic_align.core import maximum_path_c\n"
        "except ModuleNotFoundError:\n"
        "    from .core import maximum_path_c\n"
    )
mono_init.write_text(mono_src, encoding="utf-8")
PYCODE

echo "[9/11] Copiando silero_vad.onnx para o caminho esperado do piper_train"
"$PY" - <<PYCODE
from pathlib import Path
from importlib import resources
import shutil

site_pkgs = Path(r"$SITE_PKGS")
dst = site_pkgs / "piper_train" / "norm_audio" / "models" / "silero_vad.onnx"
dst.parent.mkdir(parents=True, exist_ok=True)
src = resources.files("silero_vad.data").joinpath("silero_vad.onnx")
shutil.copyfile(src, dst)
print(dst)
PYCODE

echo "[10/11] Validacoes rapidas"
"$PY" -c "import torch, pytorch_lightning as pl, piper_train; print('torch', torch.__version__, 'gpu', torch.cuda.is_available()); print('pl', pl.__version__); print('piper_train', piper_train.__file__)"
"$PY" -c "import piper_train.vits.monotonic_align as m; print('monotonic_align ok', m.__file__)"
"$PY" -c "from pathlib import Path; import piper_train; p=Path(piper_train.__file__).resolve().parent/'norm_audio'/'models'/'silero_vad.onnx'; print('silero_vad exists', p.exists(), p)"

echo "[11/11] Ambiente recriado com sucesso"
echo "Use: $PY treinar_vozes_piper.py --max-epochs-cap 2000 --max-train-hours 2 --batch-size 16 --preprocess-max-workers 16"
