import argparse
import sys
import pytest
from src.evaluation.benchmark import run_full_benchmark
from src.data.collector import NBADataCollector
from src.utils.config import ProjectPaths, ModelConfig

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NBA Match Predictor - Predição de Partidas com Redes Neurais e Métricas de Posse"
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "collect", "test"],
        default="benchmark",
        help="Modo de execução: benchmark (treino e avaliação dos 9 modelos), collect (coleta de dados da NBA) ou test (suíte de testes automatizados)."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Forçar execução em CPU ou GPU CUDA (padrão: detecção automática)."
    )
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    paths = ProjectPaths()
    config = ModelConfig()
    if args.device:
        config.device = args.device

    if args.mode == "collect":
        collector = NBADataCollector(paths=paths)
        collector.collect_all_seasons()
        print("Coleta de dados finalizada com sucesso.")
        return 0

    if args.mode == "test":
        print("Executando suíte de testes com Pytest...")
        return pytest.main(["tests/", "-v"])

    run_full_benchmark(paths=paths, config=config)
    return 0

if __name__ == "__main__":
    sys.exit(main())
