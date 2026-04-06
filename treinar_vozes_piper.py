import argparse
import csv
import json
import os
import select
import signal
import shlex
import shutil
import subprocess
import sys
import termios
import time
import urllib.request
import tty
from datetime import datetime
from pathlib import Path


def count_lines(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int((current / total) * width)
    return "[" + ("#" * filled) + ("." * (width - filled)) + "]"


def pct(current: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (current / total) * 100.0


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def format_seconds(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def tail_text_file(file_path: Path, max_lines: int = 25) -> str:
    if not file_path.exists():
        return "(log não encontrado)"

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"(erro ao ler log: {exc})"

    tail = lines[-max_lines:]
    return "".join(tail).strip() if tail else "(log vazio)"


def print_dashboard(stats: dict, ranking: list[tuple[str, int]]) -> None:
    clear_screen()
    width = max(90, shutil.get_terminal_size((100, 30)).columns)
    sep = "=" * width

    print(sep)
    print(" TREINO DE VOZES PIPER - DASHBOARD ".center(width))
    print(sep)
    print(f"Início.................: {stats['started_at']}")
    print(f"Idioma.................: {stats['language']}")
    print(f"Dataset raiz...........: {stats['dataset_root']}")
    print(f"Modelos saída..........: {stats['models_root']}")
    print("-" * width)

    print(
        f"Vozes elegíveis........: {stats['eligible_done']}/{stats['eligible_total']} "
        f"({pct(stats['eligible_done'], stats['eligible_total']):6.2f}%) "
        f"{progress_bar(stats['eligible_done'], stats['eligible_total'])}"
    )
    print(f"Voz atual..............: {stats['current_voice']}")
    print(f"Status voz.............: {stats['current_stage']} {stats.get('stage_spinner', '')}".rstrip())
    print(f"Tempo da etapa.........: {stats.get('stage_elapsed', '00:00:00')}")
    print("Controles..............: [n] próximo passo | [q] sair")
    if stats.get("last_log_file"):
        print(f"Log atual..............: {stats['last_log_file']}")
    if stats.get("last_error"):
        print(f"Último erro............: {stats['last_error']}")
    print("-" * width)

    print(f"Treinos ok.............: {stats['voices_ok']}")
    print(f"Falhas preprocess......: {stats['fail_preprocess']}")
    print(f"Falhas treino..........: {stats['fail_train']}")
    print(f"Falhas export..........: {stats['fail_export']}")
    print(f"Puladas (< min)........: {stats['voices_skipped_min']}")
    print(f"Puladas (erro dados)...: {stats['voices_skipped_bad']}")

    print("-" * width)
    print("Top 5 vozes por frases (apenas visualização):")
    top = ranking[:5]
    if not top:
        print("  (nenhuma voz encontrada)")
    else:
        for idx, (voice_id, lines) in enumerate(top, start=1):
            print(f"  {idx}. {voice_id:<18} {lines:>6} frases")
    print(sep)


def run_cmd(
    cmd: str,
    dry_run: bool,
    env: dict | None = None,
    stats: dict | None = None,
    ranking: list[tuple[str, int]] | None = None,
    log_file: Path | None = None,
    timeout_seconds: int | None = None,
    allow_ctrl_c_continue: bool = False,
) -> tuple[int, bool, bool, str | None]:
    if dry_run:
        print(f"[dry-run] {cmd}")
        return 0, False, False, None

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    spinner = ["|", "/", "-", "\\"]
    start = time.time()

    def stop_process(proc: subprocess.Popen, graceful_timeout: int = 20) -> int:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

        try:
            proc.wait(timeout=graceful_timeout)
        except KeyboardInterrupt:
            # Segundo Ctrl+C: força encerramento imediato sem traceback.
            proc.kill()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

        return proc.returncode or 130

    old_tty_mode = None
    if sys.stdin.isatty():
        try:
            fd = sys.stdin.fileno()
            old_tty_mode = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            old_tty_mode = None

    try:
        with (log_file.open("a", encoding="utf-8") if log_file else open(os.devnull, "w", encoding="utf-8")) as out:
            out.write(f"\n[{datetime.now().isoformat()}] CMD: {cmd}\n")
            out.flush()

            process = subprocess.Popen(
                cmd,
                shell=True,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
            )

            tick = 0
            while True:
                try:
                    code = process.poll()
                    elapsed = int(time.time() - start)

                    if old_tty_mode is not None:
                        try:
                            ready, _, _ = select.select([sys.stdin], [], [], 0)
                            if ready:
                                key = sys.stdin.read(1).lower()
                                if key == "n":
                                    out.write(
                                        f"[{datetime.now().isoformat()}] INFO: tecla 'n' detectada. Avançando para o próximo passo.\n"
                                    )
                                    out.flush()
                                    return stop_process(process), False, True, "next"
                                if key == "q":
                                    out.write(
                                        f"[{datetime.now().isoformat()}] INFO: tecla 'q' detectada. Encerrando execução.\n"
                                    )
                                    out.flush()
                                    return stop_process(process), False, True, "quit"
                        except Exception:
                            pass

                    if timeout_seconds is not None and elapsed >= timeout_seconds and code is None:
                        out.write(
                            f"[{datetime.now().isoformat()}] INFO: limite de tempo atingido ({timeout_seconds}s). Encerrando treino.\n"
                        )
                        out.flush()
                        return stop_process(process), True, False, None

                    if stats is not None:
                        stats["stage_elapsed"] = format_seconds(elapsed)
                        stats["stage_spinner"] = spinner[tick % len(spinner)]
                        if log_file is not None:
                            stats["last_log_file"] = str(log_file)
                        if ranking is not None:
                            print_dashboard(stats, ranking)

                    if code is not None:
                        return code, False, False, None

                    tick += 1
                    time.sleep(1)
                except KeyboardInterrupt:
                    if not allow_ctrl_c_continue:
                        return stop_process(process), False, True, "quit"

                    out.write(
                        f"[{datetime.now().isoformat()}] INFO: Ctrl+C detectado. Encerrando treino atual e seguindo para export.\n"
                    )
                    out.flush()
                    return stop_process(process), False, True, "next"
    finally:
        if old_tty_mode is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_tty_mode)
            except Exception:
                pass


def render_cmd(template: str, values: dict) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        missing = str(exc)
        raise ValueError(f"Placeholder ausente no template: {missing}") from exc


def load_dataset_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_piper_train_paths(piper_train_dir: Path) -> list[Path]:
    candidates = [
        piper_train_dir,
        piper_train_dir / "src",
        piper_train_dir / "src" / "python",
        piper_train_dir / "src" / "python_run",
        piper_train_dir / "src" / "python" / "training",
    ]
    valid = []
    for candidate in candidates:
        if (candidate / "piper_train").is_dir():
            valid.append(candidate)
    return valid


def with_env_pythonpath(base_env: dict, extra_paths: list[Path]) -> dict:
    env = dict(base_env)
    if not extra_paths:
        return env

    extra = os.pathsep.join(str(p.resolve()) for p in extra_paths)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{extra}{os.pathsep}{existing}" if existing else extra
    return env


def has_module(module_name: str, env: dict) -> bool:
    # Verifica no mesmo Python do script para evitar mismatch de ambiente.
    probe = (
        "import importlib.util, sys; "
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    result = subprocess.run([sys.executable, "-c", probe], check=False, env=env)
    return result.returncode == 0


def detect_torch_gpu(env: dict) -> tuple[bool, str]:
    code = (
        "import torch; "
        "ok=torch.cuda.is_available() and torch.cuda.device_count()>0; "
        "name=torch.cuda.get_device_name(0) if ok else 'none'; "
        "print(('OK:' + name) if ok else 'NO_GPU')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    out = (result.stdout or "").strip()
    if result.returncode == 0 and out.startswith("OK:"):
        return True, out.replace("OK:", "", 1)
    return False, out or (result.stderr or "").strip()


def detect_resume_flag(env: dict) -> str:
    help_cmd = [sys.executable, "-m", "piper_train", "--help"]
    result = subprocess.run(help_cmd, check=False, env=env, capture_output=True, text=True)
    help_text = f"{result.stdout}\n{result.stderr}"
    if "--resume_from_checkpoint" in help_text:
        return "--resume_from_checkpoint"
    if "--ckpt_path" in help_text:
        return "--ckpt_path"
    return ""


def has_silero_vad_model(env: dict) -> tuple[bool, str]:
    code = (
        "from pathlib import Path; "
        "import piper_train, sys; "
        "model = Path(piper_train.__file__).resolve().parent / 'norm_audio' / 'models' / 'silero_vad.onnx'; "
        "print(str(model)); "
        "sys.exit(0 if model.exists() else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    path = (result.stdout or "").strip()
    return result.returncode == 0, path


def ensure_silero_vad_model(env: dict) -> tuple[bool, str]:
    ok, model_path_str = has_silero_vad_model(env)
    if ok:
        return True, model_path_str

    model_path = Path(model_path_str) if model_path_str else None
    if model_path is None:
        return False, "caminho do silero_vad.onnx não pôde ser determinado"

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Fonte preferencial: pacote silero-vad instalado no mesmo ambiente.
    silero_data_path = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "silero_vad"
        / "data"
        / "silero_vad.onnx"
    )
    if silero_data_path.exists():
        try:
            shutil.copyfile(silero_data_path, model_path)
            if model_path.exists() and model_path.stat().st_size > 0:
                return True, str(model_path)
        except Exception as exc:
            return False, f"falha ao copiar silero_vad.onnx do pacote silero-vad: {exc}"

    urls = [
        "https://github.com/snakers4/silero-vad/raw/main/files/silero_vad.onnx",
        "https://raw.githubusercontent.com/snakers4/silero-vad/main/files/silero_vad.onnx",
    ]

    last_error = ""
    for url in urls:
        try:
            urllib.request.urlretrieve(url, str(model_path))
            if model_path.exists() and model_path.stat().st_size > 0:
                return True, str(model_path)
            last_error = f"download sem conteúdo válido de {url}"
        except Exception as exc:
            last_error = f"falha ao baixar de {url}: {exc}"

    return False, last_error or "falha ao obter silero_vad.onnx"


def discover_voices(dataset_root: Path) -> list[tuple[str, Path, int]]:
    voices = []
    for voice_dir in sorted(dataset_root.glob("voice_*")):
        if not voice_dir.is_dir():
            continue
        metadata = voice_dir / "metadata.csv"
        lines = count_lines(metadata)
        voices.append((voice_dir.name, voice_dir, lines))
    return voices


def find_latest_checkpoint(voice_work_dir: Path) -> Path | None:
    checkpoint_root = voice_work_dir / "lightning_logs"
    if not checkpoint_root.exists():
        return None

    # Prioriza last.ckpt dos runs mais recentes.
    last_candidates = sorted(
        checkpoint_root.glob("version_*/checkpoints/last.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if last_candidates:
        return last_candidates[0]

    # Fallback para qualquer .ckpt mais recente.
    any_candidates = sorted(
        checkpoint_root.glob("version_*/checkpoints/*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return any_candidates[0] if any_candidates else None


def save_ranking_csv(output_file: Path, ranking: list[tuple[str, int]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["voice_id", "num_phrases"])
        for voice_id, lines in ranking:
            writer.writerow([voice_id, lines])


def has_preprocessed_dataset(voice_work_dir: Path) -> bool:
    return (voice_work_dir / "dataset.jsonl").exists() and (voice_work_dir / "config.json").exists()


def log_has_timeout_marker(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "limite de tempo atingido" in line:
                    return True
    except OSError:
        return False
    return False


def checkpoint_epoch_reached(checkpoint_path: Path | None, max_epochs: int) -> bool:
    if checkpoint_path is None:
        return False

    try:
        # Carrega só metadados do checkpoint para verificar avanço de época.
        import torch  # type: ignore

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        epoch = ckpt.get("epoch")
        if isinstance(epoch, int):
            return (epoch + 1) >= max_epochs
    except Exception:
        return False

    return False


def validate_ljspeech_metadata(voice_dir: Path) -> tuple[bool, str]:
    metadata_path = voice_dir / "metadata.csv"
    if not metadata_path.exists():
        return False, f"metadata.csv ausente em {voice_dir}"

    checked = 0
    try:
        with metadata_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                checked += 1
                # Common Voice TSV vazou para metadata por voz.
                if "\t" in line and "|" not in line:
                    return False, "metadata parece TSV/Common Voice, não LJSpeech"

                parts = line.split("|")
                if len(parts) < 2:
                    return False, "linha inválida no metadata (esperado formato LJSpeech: wav|texto)"

                wav_rel = parts[0].strip()
                if not wav_rel:
                    return False, "linha inválida no metadata (arquivo wav vazio)"

                if len(wav_rel) > 240:
                    return False, "campo de arquivo wav muito longo (metadata corrompido)"

                if "\t" in wav_rel:
                    return False, "nome do wav contém TAB (metadata inválido)"

                if checked >= 200:
                    break
    except OSError as exc:
        return False, f"erro ao ler metadata: {exc}"

    if checked == 0:
        return False, "metadata.csv vazio"

    return True, ""


def preprocess_error_is_bad_data(log_tail: str) -> bool:
    markers = [
        "File name too long",
        "Errno 36",
        "metadata parece TSV",
        "ljspeech_dataset",
    ]
    lowered = log_tail.lower()
    return any(marker.lower() in lowered for marker in markers)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    argv = sys.argv[1:]

    def cli_has_flag(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)

    parser = argparse.ArgumentParser(
        description="Treina vozes Piper em lote e exporta ONNX/ONNX JSON"
    )
    parser.add_argument("--dataset-root", default="piper_dataset_por_voz")
    parser.add_argument("--work-root", default="piper_training_work")
    parser.add_argument("--models-root", default="piper_models")
    parser.add_argument("--config", default="dataset_config.json")
    parser.add_argument(
        "--min-phrases",
        type=int,
        default=0,
        help="Número mínimo de frases por voz (padrão: 0 = todas as vozes)"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument(
        "--max-epochs-cap",
        type=int,
        default=2000,
        help="Teto de épocas por voz para evitar treinos longos (0 = sem teto)"
    )
    parser.add_argument(
        "--max-train-hours",
        type=float,
        default=2.0,
        help="Tempo máximo de treino por voz em horas (0 = sem limite)"
    )
    parser.add_argument("--language", default="pt-br")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--preprocess-max-workers",
        type=int,
        default=4,
        help="Quantidade de workers para o preprocess (padrão: 4)"
    )
    parser.add_argument(
        "--only-top",
        type=int,
        default=0,
        help="Ignorado: o script sempre treina todas as vozes (ordem desc por frases)"
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permite continuar em CPU se GPU não estiver disponível"
    )
    parser.add_argument(
        "--piper-train-dir",
        default="",
        help="Caminho para repo/pasta que contenha piper_train (opcional)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preprocess-cmd",
        default=(
            "{python_exec} -m piper_train.preprocess "
            "--language {language} --input-dir {voice_dir} --output-dir {voice_work_dir} "
            "--dataset-format ljspeech --sample-rate {sample_rate} "
            "--max-workers {preprocess_max_workers}"
        ),
        help="Template comando preprocess"
    )
    parser.add_argument(
        "--train-cmd",
        default=(
            "{python_exec} -m piper_train "
            "--dataset-dir {voice_work_dir} --accelerator gpu --devices 1 "
            "--batch-size {batch_size} --max_epochs {max_epochs} "
            "--num_sanity_val_steps 0 --limit_val_batches 0 "
            "--validation-split 0 --num-test-examples 0 --precision 32 "
            "{resume_from_checkpoint_arg}"
        ),
        help="Template comando treino"
    )
    parser.add_argument(
        "--export-cmd",
        default=(
            "{python_exec} -m piper_train.export_onnx "
            "{checkpoint_path} {model_onnx_path}"
        ),
        help="Template comando export"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    work_root = Path(args.work_root).resolve()
    models_root = Path(args.models_root).resolve()
    config_path = Path(args.config).resolve()
    piper_train_dir = Path(args.piper_train_dir).resolve() if args.piper_train_dir else None

    config = load_dataset_config(config_path)
    if config:
        if not cli_has_flag("--language"):
            args.language = config.get("language", {}).get("code", args.language)
        if not cli_has_flag("--sample-rate"):
            args.sample_rate = config.get("audio", {}).get("sample_rate", args.sample_rate)
        train_cfg = config.get("training", {})
        if not cli_has_flag("--batch-size"):
            args.batch_size = train_cfg.get("batch_size", args.batch_size)
        if not cli_has_flag("--max-epochs"):
            args.max_epochs = train_cfg.get("max_epochs", args.max_epochs)

    if args.preprocess_max_workers < 1:
        args.preprocess_max_workers = 1

    if args.max_epochs_cap > 0:
        args.max_epochs = min(args.max_epochs, args.max_epochs_cap)

    if not dataset_root.exists():
        print(f"ERRO: dataset não encontrado: {dataset_root}", file=sys.stderr)
        return 2

    extra_pythonpaths: list[Path] = []
    if piper_train_dir is not None:
        if not piper_train_dir.exists():
            print(f"ERRO: --piper-train-dir não existe: {piper_train_dir}", file=sys.stderr)
            return 2
        extra_pythonpaths = find_piper_train_paths(piper_train_dir)

    command_env = with_env_pythonpath(os.environ, extra_pythonpaths)
    existing_warnings = command_env.get("PYTHONWARNINGS", "")
    warning_filter = "ignore::FutureWarning"
    command_env["PYTHONWARNINGS"] = (
        f"{existing_warnings},{warning_filter}" if existing_warnings else warning_filter
    )
    command_env["AMD_SERIALIZE_KERNEL"] = "3"
    command_env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    command_env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    command_env["HIP_VISIBLE_DEVICES"] = "0"
    command_env["TORCH_FLOAT32_MATMUL_PRECISION"] = "medium"

    piper_train_available = args.dry_run or has_module("piper_train", command_env)
    using_local_fallback = False

    # Só usa piper-src local automaticamente quando o módulo não existir no venv.
    if piper_train_dir is None and not piper_train_available:
        local_piper_src = script_dir / "piper-src"
        if local_piper_src.exists():
            piper_train_dir = local_piper_src
            extra_pythonpaths = find_piper_train_paths(piper_train_dir)
            command_env = with_env_pythonpath(os.environ, extra_pythonpaths)
            command_env["PYTHONWARNINGS"] = (
                f"{existing_warnings},{warning_filter}" if existing_warnings else warning_filter
            )
            command_env["AMD_SERIALIZE_KERNEL"] = "3"
            command_env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
            command_env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
            command_env["HIP_VISIBLE_DEVICES"] = "0"
            command_env["TORCH_FLOAT32_MATMUL_PRECISION"] = "medium"
            using_local_fallback = bool(extra_pythonpaths)
            piper_train_available = args.dry_run or has_module("piper_train", command_env)
            if piper_train_available or using_local_fallback:
                print(f"Usando piper_train local em: {piper_train_dir}")

    if not piper_train_available and not using_local_fallback:
        print("ERRO: módulo 'piper_train' não encontrado no ambiente atual.", file=sys.stderr)
        print(
            "Dica 1: instale o pacote de treino do Piper no venv ativo.",
            file=sys.stderr,
        )
        print(
            "Dica 2: se você já clonou o repositório do Piper, rode com --piper-train-dir /caminho/do/piper.",
            file=sys.stderr,
        )
        print(
            f"Python atual: {sys.executable}",
            file=sys.stderr,
        )
        if extra_pythonpaths:
            print(
                "PYTHONPATH extra detectado: "
                + ", ".join(str(p) for p in extra_pythonpaths),
                file=sys.stderr,
            )
        return 3

    if using_local_fallback and not piper_train_available:
        print(
            "AVISO: validação prévia do módulo piper_train falhou, mas o fallback local foi habilitado.",
            file=sys.stderr,
        )

    resume_flag = ""
    if piper_train_available and not args.dry_run:
        resume_flag = detect_resume_flag(command_env)
        if not resume_flag:
            print(
                "AVISO: piper_train não expõe flag de retomada de checkpoint; retomada automática ficará desabilitada.",
                file=sys.stderr,
            )

    if piper_train_available and not args.dry_run:
        vad_ok, vad_result = ensure_silero_vad_model(command_env)
        if vad_ok:
            print(f"silero_vad.onnx disponível em: {vad_result}")
        else:
            print("ERRO: não foi possível preparar silero_vad.onnx para preprocess com áudio.", file=sys.stderr)
            print(f"Detalhe: {vad_result}", file=sys.stderr)
            return 5

    if not args.dry_run:
        gpu_ok, gpu_info = detect_torch_gpu(command_env)
        if not gpu_ok and not args.allow_cpu_fallback:
            print("ERRO: GPU não detectada pelo PyTorch no ambiente atual.", file=sys.stderr)
            print("Use --allow-cpu-fallback para rodar em CPU (lento).", file=sys.stderr)
            print("Verifique ROCm/driver e o torch do venv antes de treinar.", file=sys.stderr)
            if gpu_info:
                print(f"Detalhe: {gpu_info}", file=sys.stderr)
            return 4
        if gpu_ok:
            print(f"GPU detectada para treino: {gpu_info}")
            print("Kernels HIP serão serializados via AMD_SERIALIZE_KERNEL=3 para expor falhas assíncronas.")
        else:
            print("AVISO: seguindo em CPU por causa de --allow-cpu-fallback")

    if extra_pythonpaths:
        probe_code = (
            "import piper_train; "
            "print(getattr(piper_train, '__file__', '<sem arquivo>'))"
        )
        probe_result = subprocess.run(
            [sys.executable, "-c", probe_code],
            check=False,
            env=command_env,
            capture_output=True,
            text=True,
        )
        if probe_result.returncode == 0 and probe_result.stdout.strip():
            print(f"piper_train carregado de: {probe_result.stdout.strip()}")
        else:
            print("AVISO: não foi possível confirmar o caminho do módulo piper_train.")

    if args.max_train_hours > 0:
        print(f"Limite de treino por voz: {args.max_train_hours:.2f}h, até {args.max_epochs} épocas")
    else:
        print(f"Limite de treino por voz: sem limite de tempo, até {args.max_epochs} épocas")

    voices = discover_voices(dataset_root)
    ranking = sorted([(name, lines) for name, _, lines in voices], key=lambda x: x[1], reverse=True)

    ranking_file = work_root / "voice_ranking.csv"
    save_ranking_csv(ranking_file, ranking)

    eligible = [(name, path, lines) for name, path, lines in voices if lines >= args.min_phrases]
    eligible.sort(key=lambda x: x[2], reverse=True)

    if args.only_top > 0:
        print(
            "AVISO: --only-top foi informado, mas será ignorado. "
            "Treinando todas as vozes da maior para a menor.",
            file=sys.stderr,
        )

    stats = {
        "started_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "language": args.language,
        "dataset_root": str(dataset_root),
        "models_root": str(models_root),
        "eligible_total": len(eligible),
        "eligible_done": 0,
        "current_voice": "-",
        "current_stage": "iniciando",
        "voices_ok": 0,
        "fail_preprocess": 0,
        "fail_train": 0,
        "fail_export": 0,
        "voices_skipped_min": len([1 for _, _, lines in voices if lines < args.min_phrases]),
        "voices_skipped_bad": 0,
        "stage_elapsed": "00:00:00",
        "stage_spinner": "",
        "last_log_file": "",
        "last_error": "",
    }

    print_dashboard(stats, ranking)

    if not eligible:
        print("\nNenhuma voz elegível para treino com o mínimo de frases informado.")
        print(f"Ranking salvo em: {ranking_file}")
        return 0

    for voice_name, voice_path, num_phrases in eligible:
        voice_work_dir = work_root / voice_name
        voice_model_dir = models_root / voice_name
        logs_dir = work_root / "logs"
        voice_model_dir.mkdir(parents=True, exist_ok=True)

        model_onnx_path = voice_model_dir / f"{voice_name}.onnx"

        values = {
            "voice_id": voice_name,
            "voice_dir": str(voice_path),
            "voice_work_dir": str(voice_work_dir),
            "voice_model_dir": str(voice_model_dir),
            "model_onnx_path": str(model_onnx_path),
            "python_exec": shlex.quote(sys.executable),
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "language": args.language,
            "sample_rate": args.sample_rate,
            "preprocess_max_workers": args.preprocess_max_workers,
            "num_phrases": num_phrases,
        }

        stats["current_voice"] = f"{voice_name} ({num_phrases} frases)"
        stats["last_error"] = ""
        stats["current_stage"] = "preprocess"
        stats["stage_elapsed"] = "00:00:00"
        stats["stage_spinner"] = ""
        stats["last_log_file"] = str(logs_dir / f"{voice_name}_preprocess.log")
        print_dashboard(stats, ranking)

        preprocess_log = logs_dir / f"{voice_name}_preprocess.log"
        preprocess_ready = args.dry_run or has_preprocessed_dataset(voice_work_dir)
        if preprocess_ready:
            stats["current_stage"] = "preprocess (reutilizado)"
            stats["stage_elapsed"] = "00:00:00"
            stats["stage_spinner"] = ""
            print_dashboard(stats, ranking)
            print(f"AVISO: preprocess existente para {voice_name}; pulando preprocess.")
        else:
            metadata_ok, metadata_reason = validate_ljspeech_metadata(voice_path)
            if not metadata_ok:
                stats["voices_skipped_bad"] += 1
                stats["last_error"] = f"dados inválidos em {voice_name}: {metadata_reason}"
                stats["current_stage"] = "pulada (dados inválidos)"
                stats["eligible_done"] += 1
                print_dashboard(stats, ranking)
                print(f"\nPULADA {voice_name}: {metadata_reason}")
                continue

            preprocess_cmd = render_cmd(args.preprocess_cmd, values)
            preprocess_code, _, preprocess_interrupted, preprocess_action = run_cmd(
                preprocess_cmd,
                args.dry_run,
                env=command_env,
                stats=stats,
                ranking=ranking,
                log_file=preprocess_log,
            )
            if preprocess_interrupted:
                if preprocess_action == "next":
                    stats["last_error"] = "preprocess pulado pelo usuário (tecla n)"
                    stats["eligible_done"] += 1
                    print_dashboard(stats, ranking)
                    print(f"\nPULADA {voice_name}: preprocess interrompido via tecla n.")
                    continue
                stats["last_error"] = f"preprocess interrompido pelo usuário (código {preprocess_code})"
                print_dashboard(stats, ranking)
                print(f"\nPROCESSO INTERROMPIDO durante preprocess de {voice_name} (código {preprocess_code})")
                print(tail_text_file(preprocess_log))
                return preprocess_code

            if preprocess_code != 0:
                tail = tail_text_file(preprocess_log)
                if preprocess_error_is_bad_data(tail):
                    stats["voices_skipped_bad"] += 1
                    stats["last_error"] = (
                        f"preprocess inválido (dados) (código {preprocess_code})\n"
                        f"{tail}"
                    )
                    stats["current_stage"] = "pulada (dados inválidos)"
                else:
                    stats["fail_preprocess"] += 1
                    stats["last_error"] = (
                        f"preprocess falhou (código {preprocess_code})\n"
                        f"{tail}"
                    )
                stats["last_error"] = (
                    stats["last_error"]
                )
                stats["eligible_done"] += 1
                print_dashboard(stats, ranking)
                print(f"\nFALHA EM preprocess para {voice_name} (código {preprocess_code})")
                print(tail)
                continue

        stats["current_stage"] = "treino"
        stats["stage_elapsed"] = "00:00:00"
        stats["stage_spinner"] = ""
        stats["last_log_file"] = str(logs_dir / f"{voice_name}_train.log")
        print_dashboard(stats, ranking)

        train_log = logs_dir / f"{voice_name}_train.log"
        checkpoint_before_train = None if args.dry_run else find_latest_checkpoint(voice_work_dir)
        prev_timed_out = log_has_timeout_marker(train_log)
        prev_reached_epochs = checkpoint_epoch_reached(checkpoint_before_train, args.max_epochs)

        train_timed_out = False
        train_interrupted = False
        if checkpoint_before_train is not None and (prev_timed_out or prev_reached_epochs):
            stats["current_stage"] = "treino (reutilizado)"
            stats["stage_elapsed"] = "00:00:00"
            stats["stage_spinner"] = ""
            print_dashboard(stats, ranking)
            reason = "tempo limite anterior" if prev_timed_out else "épocas já atingidas"
            print(
                f"AVISO: treino anterior de {voice_name} já atingiu {reason}; pulando treino e seguindo para export."
            )
        else:
            values["resume_from_checkpoint_arg"] = (
                f"{resume_flag} {shlex.quote(str(checkpoint_before_train))}"
                if checkpoint_before_train is not None and resume_flag
                else ""
            )
            train_cmd = render_cmd(args.train_cmd, values)
            train_timeout = int(args.max_train_hours * 3600) if args.max_train_hours > 0 else None
            train_code, train_timed_out, train_interrupted, train_action = run_cmd(
                train_cmd,
                args.dry_run,
                env=command_env,
                stats=stats,
                ranking=ranking,
                log_file=train_log,
                timeout_seconds=train_timeout,
                allow_ctrl_c_continue=True,
            )
            if train_interrupted and train_action == "quit":
                stats["last_error"] = f"treino interrompido pelo usuário (código {train_code})"
                print_dashboard(stats, ranking)
                print(f"\nPROCESSO INTERROMPIDO durante treino de {voice_name} (código {train_code})")
                print(tail_text_file(train_log))
                return train_code

            if train_code != 0 and not train_timed_out and not train_interrupted:
                stats["fail_train"] += 1
                stats["last_error"] = (
                    f"treino falhou (código {train_code})\n"
                    f"{tail_text_file(train_log)}"
                )
                stats["eligible_done"] += 1
                print_dashboard(stats, ranking)
                print(f"\nFALHA EM treino para {voice_name} (código {train_code})")
                print(tail_text_file(train_log))
                continue

            if train_timed_out:
                print(f"AVISO: treino de {voice_name} atingiu limite de tempo; tentando exportar checkpoint mais recente.")
            if train_interrupted:
                print(f"AVISO: treino de {voice_name} interrompido por Ctrl+C; tentando exportar checkpoint mais recente.")

        stats["current_stage"] = "export"
        stats["stage_elapsed"] = "00:00:00"
        stats["stage_spinner"] = ""
        stats["last_log_file"] = str(logs_dir / f"{voice_name}_export.log")
        print_dashboard(stats, ranking)

        if args.dry_run:
            values["checkpoint_path"] = str(
                voice_work_dir / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt"
            )
        else:
            checkpoint_path = find_latest_checkpoint(voice_work_dir)
            if checkpoint_path is None:
                stats["fail_export"] += 1
                stats["last_error"] = f"checkpoint não encontrado em {voice_work_dir}"
                stats["eligible_done"] += 1
                print(f"ERRO: checkpoint não encontrado para {voice_name} em {voice_work_dir}")
                print_dashboard(stats, ranking)
                continue

            values["checkpoint_path"] = str(checkpoint_path)

        export_cmd = render_cmd(args.export_cmd, values)
        export_log = logs_dir / f"{voice_name}_export.log"
        export_code, _, export_interrupted, export_action = run_cmd(
            export_cmd,
            args.dry_run,
            env=command_env,
            stats=stats,
            ranking=ranking,
            log_file=export_log,
        )
        if export_interrupted:
            if export_action == "next":
                stats["last_error"] = "export pulado pelo usuário (tecla n)"
                stats["eligible_done"] += 1
                print_dashboard(stats, ranking)
                print(f"\nPULADA {voice_name}: export interrompido via tecla n.")
                continue

            stats["last_error"] = f"export interrompido pelo usuário (código {export_code})"
            print_dashboard(stats, ranking)
            print(f"\nPROCESSO INTERROMPIDO durante export de {voice_name} (código {export_code})")
            print(tail_text_file(export_log))
            return export_code

        if export_code != 0:
            stats["fail_export"] += 1
            stats["last_error"] = (
                f"export falhou (código {export_code})\n"
                f"{tail_text_file(export_log)}"
            )
            stats["eligible_done"] += 1
            print_dashboard(stats, ranking)
            print(f"\nFALHA EM export para {voice_name} (código {export_code})")
            print(tail_text_file(export_log))
            continue

        if not args.dry_run:
            config_json = voice_work_dir / "config.json"
            model_config_json = Path(str(model_onnx_path) + ".json")
            if config_json.exists():
                shutil.copyfile(config_json, model_config_json)
            else:
                print(f"AVISO: config.json não encontrado para gerar {model_config_json}")

        stats["voices_ok"] += 1
        stats["eligible_done"] += 1
        stats["current_stage"] = "concluído"
        stats["last_error"] = ""
        print_dashboard(stats, ranking)

    print("\nProcesso finalizado.")
    print(f"Ranking salvo em: {ranking_file}")
    print(f"Modelos em: {models_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
