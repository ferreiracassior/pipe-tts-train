#!/usr/bin/env bash
set -euo pipefail

# Recria o ambiente Python do projeto pipe-tts-train
# sem manter piper-src dentro do workspace.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="python3.12"
REQUIRED_PYTHON_VERSION="3.12.13"

# Versoes fixadas (baseline atual)
PIP_VERSION="26.0.1"
SETUPTOOLS_VERSION="70.2.0"
WHEEL_VERSION="0.46.3"

TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.1"
TORCH_VERSION="2.6.0+rocm6.1"
TORCHVISION_VERSION="0.21.0+rocm6.1"
TORCHAUDIO_VERSION="2.6.0+rocm6.1"

PYTORCH_LIGHTNING_VERSION="1.9.5"
ONNXRUNTIME_VERSION="1.24.4"
LIBROSA_VERSION="0.11.0"
NUMPY_VERSION="2.4.3"
SCIPY_VERSION="1.17.1"
SCIKIT_LEARN_VERSION="1.8.0"
NUMBA_VERSION="0.65.0"
CYTHON_VERSION="0.29.37"
SOUNDFILE_VERSION="0.13.1"
SILERO_VAD_VERSION="6.2.1"

PIPER_PHONEMIZE_COMMIT="ba3cc06c5248215928821f1393b2b854a936991a"
PIPER_REPO_URL="https://github.com/rhasspy/piper.git"
PIPER_COMMIT="73c04d81d5590ecc46e522de3601ce7fb29fc2be"

check_system_prerequisites() {
  echo "[1/12] Verificando pre-requisitos de sistema"

  local required_bins=("$PYTHON_BIN" git gcc g++ make)
  local missing=0

  for bin in "${required_bins[@]}"; do
    if ! command -v "$bin" >/dev/null 2>&1; then
      echo "ERRO: comando obrigatorio nao encontrado: $bin"
      missing=1
    fi
  done

  local python_version
  python_version="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  if [ "$python_version" != "$REQUIRED_PYTHON_VERSION" ]; then
    echo "ERRO: versao de Python incompativel. Esperado: $REQUIRED_PYTHON_VERSION, encontrado: $python_version"
    exit 1
  fi

  if [ "$missing" -ne 0 ]; then
    echo "Instale os pre-requisitos de sistema e rode novamente."
    exit 1
  fi

  "$PYTHON_BIN" - <<'PYCODE'
import ctypes.util
import sys

missing = []
for lib in ("espeak-ng", "onnxruntime"):
    if not ctypes.util.find_library(lib):
        missing.append(lib)

if missing:
    print("ERRO: bibliotecas de sistema ausentes: " + ", ".join(missing))
    print("Instale as libs no host e rode novamente.")
    sys.exit(1)
PYCODE

  echo "Pre-requisitos de sistema OK"
}

check_system_prerequisites

echo "[2/12] Criando venv em $VENV_DIR"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

echo "[3/12] Atualizando ferramentas base"
"$PIP" install --upgrade \
  "pip==$PIP_VERSION" \
  "setuptools==$SETUPTOOLS_VERSION" \
  "wheel==$WHEEL_VERSION"

echo "[4/12] Instalando stack PyTorch ROCm"
"$PIP" install --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

echo "[5/12] Instalando dependencias gerais"
"$PIP" install \
  "pytorch-lightning==$PYTORCH_LIGHTNING_VERSION" \
  "onnxruntime==$ONNXRUNTIME_VERSION" \
  "librosa==$LIBROSA_VERSION" \
  "numpy==$NUMPY_VERSION" \
  "scipy==$SCIPY_VERSION" \
  "scikit-learn==$SCIKIT_LEARN_VERSION" \
  "numba==$NUMBA_VERSION" \
  "Cython==$CYTHON_VERSION" \
  "soundfile==$SOUNDFILE_VERSION" \
  "silero-vad==$SILERO_VAD_VERSION"

echo "[6/12] Instalando piper_phonemize (fixado)"
"$PIP" install "piper_phonemize @ git+https://github.com/rhasspy/piper-phonemize.git@$PIPER_PHONEMIZE_COMMIT"

echo "[7/12] Baixando piper temporariamente para instalar piper_train"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone "$PIPER_REPO_URL" "$TMP_DIR/piper"
( cd "$TMP_DIR/piper" && git checkout "$PIPER_COMMIT" )

# Instala piper_train no venv sem deps (deps ja foram controladas acima)
"$PIP" install --no-deps "$TMP_DIR/piper/src/python"

echo "[8/12] Compilando monotonic_align"
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

echo "[9/12] Aplicando patch de compatibilidade no piper_train"
"$PY" - <<PYCODE
from pathlib import Path

site_pkgs = Path(r"$SITE_PKGS")
main_py = site_pkgs / "piper_train" / "__main__.py"
mono_init = site_pkgs / "piper_train" / "vits" / "monotonic_align" / "__init__.py"
vad_py = site_pkgs / "piper_train" / "norm_audio" / "vad.py"

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

  vad_src = vad_py.read_text(encoding="utf-8")
  if '"h0": self._h' in vad_src and '"state": self._state' not in vad_src:
    vad_src = vad_src.replace(
      "        self._h = np.zeros((2, 1, 64)).astype(\"float32\")\n"
      "        self._c = np.zeros((2, 1, 64)).astype(\"float32\")\n",
      "        input_names = {inp.name for inp in self.session.get_inputs()}\n"
      "        self._uses_state_input = (\"state\" in input_names) and (\"sr\" in input_names)\n"
      "\n"
      "        if self._uses_state_input:\n"
      "            # Newer Silero ONNX expects recurrent state + sample rate.\n"
      "            self._state = np.zeros((2, 1, 128), dtype=np.float32)\n"
      "        else:\n"
      "            # Legacy Silero ONNX expects h0/c0 inputs.\n"
      "            self._h = np.zeros((2, 1, 64), dtype=np.float32)\n"
      "            self._c = np.zeros((2, 1, 64), dtype=np.float32)\n"
    )

    vad_src = vad_src.replace(
      "        ort_inputs = {\n"
      "            \"input\": audio_array.astype(np.float32),\n"
      "            \"h0\": self._h,\n"
      "            \"c0\": self._c,\n"
      "        }\n"
      "        ort_outs = self.session.run(None, ort_inputs)\n"
      "        out, self._h, self._c = ort_outs\n"
      "\n"
      "        out = out.squeeze(2)[:, 1]  # make output type match JIT analog\n"
      "\n"
      "        return out\n",
      "        if self._uses_state_input:\n"
      "            ort_inputs = {\n"
      "                \"input\": audio_array.astype(np.float32),\n"
      "                \"state\": self._state,\n"
      "                \"sr\": np.array(sample_rate, dtype=np.int64),\n"
      "            }\n"
      "            out, self._state = self.session.run(None, ort_inputs)\n"
      "            return float(np.asarray(out).squeeze())\n"
      "\n"
      "        ort_inputs = {\n"
      "            \"input\": audio_array.astype(np.float32),\n"
      "            \"h0\": self._h,\n"
      "            \"c0\": self._c,\n"
      "        }\n"
      "        out, self._h, self._c = self.session.run(None, ort_inputs)\n"
      "        out = out.squeeze(2)[:, 1]  # make output type match JIT analog\n"
      "        return float(np.asarray(out).squeeze())\n"
    )

  vad_py.write_text(vad_src, encoding="utf-8")
PYCODE

echo "[10/12] Copiando silero_vad.onnx para o caminho esperado do piper_train"
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

echo "[11/12] Validacoes rapidas"
"$PY" -c "import torch, pytorch_lightning as pl, piper_train; print('torch', torch.__version__, 'gpu', torch.cuda.is_available()); print('pl', pl.__version__); print('piper_train', piper_train.__file__)"
"$PY" -c "import piper_train.vits.monotonic_align as m; print('monotonic_align ok', m.__file__)"
"$PY" -c "from pathlib import Path; import piper_train; p=Path(piper_train.__file__).resolve().parent/'norm_audio'/'models'/'silero_vad.onnx'; print('silero_vad exists', p.exists(), p)"

echo "[12/12] Ambiente recriado com sucesso"
echo "Use: $PY treinar_vozes_piper.py --max-epochs-cap 2000 --max-train-hours 2 --batch-size 16 --preprocess-max-workers 16"
