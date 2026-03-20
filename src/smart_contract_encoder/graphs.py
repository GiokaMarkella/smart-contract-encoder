#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from smart_contract_encoder.data_curation import *
from smart_contract_encoder.load_data import *
import json
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import glob
import itertools
tqdm.pandas()


def count_words_clean(text):
    text = re.sub(r'\\\\.*?\n', '', text)
    text = text = re.sub(r'[()_{}[\]]', ' ', text)
    return len(text.split())

def dist_graphs():
    df = load_dataset(file_type="merged", split="train")
    print(df)
    plt.rcParams.update({'font.size': 15})

    df1 = df[['code']].copy()
    df1['Dataset'] = 'NHLD'
    df1['word_count'] = df1['code'].apply(count_words_clean)

    df2 = df[['tac_code']].copy()
    df2.rename({'tac_code':'code'})
    df2['Dataset'] = 'TAC'
    df2['word_count'] = df2['tac_code'].apply(count_words_clean)

    df_combined = pd.concat([df1, df2], ignore_index=True)

    bin_edges = range(0, 6000, 200)

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=df_combined,
        x='word_count',
        hue='Dataset',
        bins=bin_edges,
        element='bars',
        stat='count',
        common_norm=False
    )

    plt.draw()
    hatches = ['//', '.']
    for container, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles[::-1]):

        handle.set_hatch(hatch)

        for rectangle in container:

            rectangle.set_hatch(hatch)
    plt.xlabel('# of Words')
    plt.ylabel('# of Functions')
    plt.tight_layout()
    plt.show()

    plt.savefig(f"dist.png", dpi=300)


def eval_graphs():
    results_dir = "./results"

    # Metrics to plot
    metrics = [
        'accuracy',
        'precision',
        'recall',
        'map',
        'mrr',
        'ndcg'
    ]
    model_renaming = {
        'sentence_encoder_untrained_func_code' : 'Source (Baseline)',
        'sentence_encoder_finetuned_tac_code': 'TAC Finetuned',
        'sentence_encoder_untrained_tac_code': 'TAC',
        'sentence_encoder_untrained_code': 'NHLD',
        'sentence_encoder_finetuned_code': 'NHLD Finetuned (LiftLM)',
        'ngram_untrained_opcode': 'Ngram Opcode',
        'smartembed_untrained_func_code': 'SmartEmbed',
        'codebert_untrained_func_code': 'CodeBERT',
        'coderankeembed_untrained_func_code': 'CodeRankEmbed'
    }

    k_values = list(range(5, 110, 10))

    all_results = {}

    for json_file in glob.glob(os.path.join(results_dir, "*.json")):
        model_name = os.path.basename(json_file).replace("_results.json", "")
        model_name = model_renaming.get(model_name)
        if model_name is None:
            continue
        with open(json_file, 'r') as file:
            data = json.load(file)
            all_results[model_name] = {metric: [data[metric][str(k)] for k in k_values] for metric in metrics}

    for metric in metrics:
        linestyles = itertools.cycle(['-', '--', '-.', ':'])
        markerkstyles = itertools.cycle(['o', 'v', "s", "<","*", ">"])
        plt.figure(figsize=(8, 5))
        plt.rcParams.update({'font.size': 10})
        names = ['Source (Baseline)', 'NHLD Finetuned (LiftLM)', 'NHLD', 'Ngram Opcode', 'TAC Finetuned', 'TAC', 'SmartEmbed', 'CodeBERT', 'CodeRankEmbed']
        for name in names:
            results = all_results[name]
            plt.plot(k_values, results[metric], marker=next(markerkstyles), linestyle=next(linestyles), label=name)

        plt.xlabel("k")
        plt.ylabel(f"{metric.capitalize()}@k")
        plt.xticks(k_values, rotation=45)
        plt.ylim(bottom=0)
        plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_at_k.png", dpi=300)




