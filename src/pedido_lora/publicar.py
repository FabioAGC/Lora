"""Publica o adaptador e o dataset no Hugging Face.

Uso:
    pip install huggingface_hub
    export HF_TOKEN=...   # token com permissão de escrita
    python -m pedido_lora.publicar --usuario SEU_USUARIO
"""
from __future__ import annotations

import argparse


def main() -> None:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--usuario", required=True)
    ap.add_argument("--adaptador", default="outputs/sopa_gulosa")
    args = ap.parse_args()

    api = HfApi()
    modelo = f"{args.usuario}/pedido-delivery-ptbr-lora"
    dados = f"{args.usuario}/pedidos-delivery-ptbr"
    api.create_repo(modelo, exist_ok=True)
    api.upload_folder(folder_path=args.adaptador, repo_id=modelo)
    api.upload_file(path_or_fileobj="docs/MODEL_CARD.md", path_in_repo="README.md", repo_id=modelo)
    api.create_repo(dados, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path="data", repo_id=dados, repo_type="dataset")
    print(f"https://huggingface.co/{modelo}\nhttps://huggingface.co/datasets/{dados}")


if __name__ == "__main__":
    main()
