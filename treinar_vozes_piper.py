import argparse
import csv
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
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
) -> tuple[int, bool, bool]:
    if dry_run:
        print(f"[dry-run] {cmd}")
        return 0, False, False

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    spinner = ["|", "/", "-", "\\"]
    start = time.time()

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

                if timeout_seconds is not None and elapsed >= timeout_seconds and code is None:
                    out.write(
                        f"[{datetime.now().isoformat()}] INFO: limite de tempo atingido ({timeout_seconds}s). Encerrando treino.\n"
                    )
                    out.flush()
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    return process.returncode or 130, True, False

                if stats is not None:
                    stats["stage_elapsed"] = format_seconds(elapsed)
                    stats["stage_spinner"] = spinner[tick % len(spinner)]
                    if log_file is not None:
                        stats["last_log_file"] = str(log_file)
                    if ranking is not None:
                        print_dashboard(stats, ranking)

                if code is not None:
                    return code, False, False

                tick += 1
                time.sleep(1)
            except KeyboardInterrupt:
                if not allow_ctrl_c_continue:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    return process.returncode or 130, False, True

                out.write(
                    f"[{datetime.now().isoformat()}] INFO: Ctrl+C detectado. Encerrando treino atual e seguindo para export.\n"
                )
                out.flush()
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                return process.returncode or 130, False, True


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


def main() -> int:
    script_dir = Path(__file__).resolve().parent
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
            "--gpus 1 --batch-size {batch_size} --max_epochs {max_epochs} "
            "--num_sanity_val_steps 0 --limit_val_batches 0 "
            "--validation-split 0 --num-test-examples 0 --precision 16"
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
        args.language = config.get("language", {}).get("code", args.language)
        args.sample_rate = config.get("audio", {}).get("sample_rate", args.sample_rate)
        train_cfg = config.get("training", {})
        args.batch_size = train_cfg.get("batch_size", args.batch_size)
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

    if not args.dry_run and not has_module("piper_train", command_env):
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

    if piper_train_dir is None:
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
            print(f"Usando piper_train local em: {piper_train_dir}")

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
        print_dashboard(stats, ranking)

        preprocess_cmd = render_cmd(args.preprocess_cmd, values)
        preprocess_log = logs_dir / f"{voice_name}_preprocess.log"
        preprocess_code, preprocess_timed_out, preprocess_interrupted = run_cmd(
            preprocess_cmd,
            args.dry_run,
            env=command_env,
            stats=stats,
            ranking=ranking,
            log_file=preprocess_log,
        )
        if preprocess_interrupted:
            stats["last_error"] = f"preprocess interrompido pelo usuário (código {preprocess_code})"
            print_dashboard(stats, ranking)
            print(f"\nPROCESSO INTERROMPIDO durante preprocess de {voice_name} (código {preprocess_code})")
            print(tail_text_file(preprocess_log))
            return preprocess_code

        if preprocess_code != 0:
            stats["fail_preprocess"] += 1
            stats["last_error"] = (
                f"preprocess falhou (código {preprocess_code})\n"
                f"{tail_text_file(preprocess_log)}"
            )
            stats["eligible_done"] += 1
            print_dashboard(stats, ranking)
            print(f"\nFALHA EM preprocess para {voice_name} (código {preprocess_code})")
            print(tail_text_file(preprocess_log))
            continue

        stats["current_stage"] = "treino"
        stats["stage_elapsed"] = "00:00:00"
        stats["stage_spinner"] = ""
        print_dashboard(stats, ranking)

        train_cmd = render_cmd(args.train_cmd, values)
        train_log = logs_dir / f"{voice_name}_train.log"
        train_timeout = int(args.max_train_hours * 3600) if args.max_train_hours > 0 else None
        train_code, train_timed_out, train_interrupted = run_cmd(
            train_cmd,
            args.dry_run,
            env=command_env,
            stats=stats,
            ranking=ranking,
            log_file=train_log,
            timeout_seconds=train_timeout,
            allow_ctrl_c_continue=True,
        )
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
        export_code, _, _ = run_cmd(
            export_cmd,
            args.dry_run,
            env=command_env,
            stats=stats,
            ranking=ranking,
            log_file=export_log,
        )
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
