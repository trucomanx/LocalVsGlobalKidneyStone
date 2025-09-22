#!/usr/bin/python3

import zipfile
import json
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted  # <- para ordenação natural
import matplotlib.pyplot as plt
import math
import os
import argparse  # <-- NOVO


def binary_entropy(p):
    if p == 0 or p == 1:  # casos degenerados
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    

def get_dataset_dict(input_dir):

    zip_folder = Path(input_dir)
    # Inicializa o dicionário final
    datasets_info = {}

    # Lista e ordena os zips naturalmente
    zip_files = natsorted(list(zip_folder.glob("*.zip")))

    # Itera com barra de progresso
    for zip_path in tqdm(zip_files, desc="Lendo ZIPs"):
        dataset_name = zip_path.stem  # Ex: dataset-96
        

        with zipfile.ZipFile(zip_path, 'r') as z:
            # Caminho do JSON dentro do zip
            json_path = f"data/dataset-article/{dataset_name}/report.json"

            if json_path in z.namelist():
                with z.open(json_path) as f:
                    data = json.load(f)
                    my_key = int(dataset_name.split("-")[1])
                    datasets_info[my_key] = data
            else:
                print(f"Atenção: {json_path} não encontrado no {zip_path.name}")

    return datasets_info


def plot_percentage_with_stone(datasets_info, output_dir, output_pdf="percentage_with_stone.pdf"):
    dataset_names = list(datasets_info.keys())
    percentages = [
        (v["with-stone"]*1.0 / (v["with-stone"] + v["without-stone"])) * 100
        for v in datasets_info.values()
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(dataset_names, percentages, color="royalblue")
    plt.xticks(rotation=90)
    plt.xlabel("patch size")
    plt.ylabel("Percent (%)")
    plt.minorticks_on()
    plt.grid(True, color="lightgray", alpha=0.7, linestyle="-", linewidth=0.7, which="major")
    plt.grid(True, color="lightgray", alpha=0.5, linestyle="--", linewidth=0.7, which="minor")
    plt.title("Percent with-stone over total")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,output_pdf))
    plt.close()
    print(f"📊 Generated: {output_pdf}")

def plot_entropy(datasets_info, output_dir, output_pdf="entropy.pdf"):
    dataset_names = list(datasets_info.keys())
    percentages = [
        binary_entropy(v["with-stone"]*1.0 / (v["with-stone"] + v["without-stone"])) 
        for v in datasets_info.values()
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(dataset_names, percentages, color="royalblue")
    plt.xticks(rotation=90)
    plt.xlabel("patch size")
    plt.ylabel("Binary entropy")
    plt.grid(True, color="lightgray", alpha=0.7, linestyle="-", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,output_pdf))
    plt.close()
    print(f"📊 Generated: {output_pdf}")


def plot_total_cases(datasets_info, output_dir, output_pdf="total_cases.pdf"):
    dataset_names = list(datasets_info.keys())
    totals = [
        v["with-stone"] + v["without-stone"]
        for v in datasets_info.values()
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(dataset_names, totals, color="seagreen")
    plt.xticks(rotation=90)
    plt.xlabel("patch size")
    plt.ylabel("Total of images")
    #plt.title("Total de WITH-STONE + WITHOUT-STONE por dataset")
    plt.grid(True, color="lightgray", alpha=0.7, linestyle="--", linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,output_pdf))
    plt.close()
    print(f"📊 Generated: {output_pdf}")

def plot_entropy_and_total( datasets_info, 
                            output_dir, 
                            output_pdf="entropy_and_total.pdf", 
                            color_entropy="black",
                            color_total="seagreen"):
    dataset_names = list(datasets_info.keys())

    # Valores de entropia
    entropies = [
        binary_entropy(v["with-stone"] / (v["with-stone"] + v["without-stone"]))
        for v in datasets_info.values()
    ]

    # Totais de imagens
    totals = [
        v["with-stone"] + v["without-stone"]
        for v in datasets_info.values()
    ]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Primeiro eixo Y (esquerda) → entropy
    ax1.bar(dataset_names, entropies, color=color_entropy, label="Entropy")
    ax1.set_xlabel("Patch size")
    ax1.set_ylabel("Binary entropy", color=color_entropy)
    ax1.tick_params(axis="y", labelcolor=color_entropy)

    # Segundo eixo Y (direita) → total images
    ax2 = ax1.twinx()
    ax2.plot(dataset_names, totals, color=color_total, marker="o", linewidth=2, label="Total images")
    ax2.set_ylabel("Total images", color=color_total)
    ax2.tick_params(axis="y", labelcolor=color_total)

    # Estética
    ax1.set_xticks(dataset_names)
    ax1.set_xticklabels(dataset_names, rotation=90)
    ax1.grid(True, color="lightgray", alpha=0.7, linestyle="--", linewidth=0.7)

    # Título
    plt.title("Entropy (left axis) vs Total Images (right axis)")

    # Ajusta layout e salva
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, output_pdf))
    plt.close()
    print(f"📊 Generated: {output_pdf}")


def dataset_report(input_dir,output_dir):
    datasets_info = get_dataset_dict(input_dir)

    # Salva o dicionário em JSON
    output_json = os.path.join(output_dir, "datasets_info.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(datasets_info, f, indent=4, ensure_ascii=False)
    print(f"💾 Saved: {output_json}")

    # Plots
    plot_percentage_with_stone(datasets_info,output_dir)
    #plot_total_cases(datasets_info,output_dir)
    #plot_entropy(datasets_info,output_dir)
    plot_entropy_and_total(datasets_info, output_dir)
    
if __name__ == "__main__":
    # Pasta onde estão os zips
    parser = argparse.ArgumentParser(description="Generate ZIP Dataset Reports (with graphics).")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory with zip files")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the PDFs")
    args = parser.parse_args()

    dataset_report(args.input_dir, args.output_dir)

