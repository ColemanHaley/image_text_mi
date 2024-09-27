import argparse
import itertools
import os
import sys

# import opinionated
# from opinionated.core import download_googlefont
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from iso639 import Lang
from scipy.stats import kendalltau, spearmanr, ttest_ind


plt.style.use("src/opinionated_ch.mplstyle")


# download_googlefont("EB Garamond", add_to_cache=True)
def plot_format():
    plt.rcParams["pgf.rcfonts"] = False
    # plt.style.use(
    #     "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
    # )
    # plt.rcParams.update({"font.family": "serif"})
    # plt.style.use("opinionated_rc")
    plt.style.use("src/opinionated_ch.mplstyle")

    # plt.rc("font", family="EB Garamond")

    # plt.rcParams["pgf.preamble"] = "\n".join(
    #     [
    #         r"\usepackage[dvipsnames]{xcolor}",
    #         r"\usepackage{mathspec}",
    #         # r'\usepackage[no-math]{fontspec}',
    #         r"\setsansfont{EB Garamond}",
    #         r"\setallmainfonts{EB Garamond}",
    #         r"\color{white}",
    #         # r"\setmathsfont(Digits,Latin){Charis SIL Compact}",
    #     ]
    # )


def plot_mi(df, lang, dataset):
    plot_format()
    sns.displot(df, x="mutual_information", kind="kde")
    plt.title(f"{lang} {dataset} - Mutual Information")
    plt.xlabel("Mutual Information")
    plt.ylabel("Density")
    plt.savefig(f"figures/mi_{lang}_{dataset}.png", backend="pgf", bbox_inches="tight")


def plot_pos(df, lang, dataset, model, ax=None):
    plot_format()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
        ax.set_title(
            f"Relative Groundedness of POS in {Lang(lang).name} for {dataset} dataset"
        )
    else:
        fig = None
        ax.set_title(f"{Lang(lang).name}")

    df.fixit = df.word_stanza + "#" + df.POS
    category_counts = df.fixit.value_counts()
    print(f"LANG: {lang}")

    # Set a threshold for the minimum frequency required to keep a category
    threshold = 1

    # Filter categories based on frequency
    valid_categories = category_counts[category_counts >= threshold].index

    # Filter the DataFrame based on valid categories
    filtered_df = df[df.fixit.isin(valid_categories)]

    filtered_df = filtered_df[
        filtered_df.POS.isin(
            [
                "AUX",
                "PART",
                "PRON",
                "SCONJ",
                "CCONJ",
                "DET",
                "ADP",
                "ADV",
                "NUM",
                "VERB",
                "ADJ",
                "NOUN",
                "PROPN",
            ]
        )
    ]

    df_fig = filtered_df[
        filtered_df.POS.isin(
            [
                "AUX",
                "PART",
                "PRON",
                "SCONJ",
                "CCONJ",
                "DET",
                "ADP",
                "ADV",
                "NUM",
                "VERB",
                "ADJ",
                "NOUN",
                "PROPN",
            ]
        )
    ]
    print(filtered_df)
    # df_fig = df_fig.groupby(["word_stanza", "POS"]).mean().reset_index()
    group_means = (
        df_fig.groupby(["POS"])["mutual_information"].mean().sort_values(ascending=True)
    )
    print(group_means)
    # sns.set(palette="Set3")
    ax.axhline(y=0, color="black")
    sns.violinplot(
        data=df_fig,
        x="POS",
        y="mutual_information",
        order=group_means.index,
        ax=ax,
        cut=0,
        hue="POS",
        density_norm="area",
        cmap="Set3",
    )
    # plot a horizontal line on the x-axis
    # ax.tick_params(labelsize="large")
    # ax.set_xlabel(
    #     "Part of Speech"
    # )  # "$\mathrm{var}(\Delta_{\\mathrm{distribution}})$")
    # ax.set_ylabel("Groundedness")
    # plt.savefig(f"figures/pos_{lang}_{dataset}.png", bbox_inches="tight")
    ax.set_xlabel(
        "Part of Speech"
    )  # "$\mathrm{var}(\Delta_{\\mathrm{distribution}})$")
    ax.set_ylabel("Groundedness")
    by_sent = df_fig.groupby("sentence")["mutual_information"].mean()
    # get mean value for sent
    print(by_sent.mean(), by_sent.std())

    stds = (
        df_fig.groupby(["POS"])["mutual_information"].std().sort_values(ascending=True)
    )
    stds = stds.reindex(group_means.index)
    print(stds)
    # ax.errorbar(
    #     group_means.index,
    #     group_means,
    #     yerr=stds,
    #     fmt="o",
    #     capsize=5,
    #     color="skyblue",
    #     markersize=8,
    #     elinewidth=2,
    #     markeredgecolor="blue",
    #     markerfacecolor="white",
    # )
    # plt.axhline(y=0)
    ax.tick_params(labelsize="large")

    if fig is not None:
        fig.savefig(f"figures/{dataset}/{model}/pos_{lang}.png", bbox_inches="tight")
    return group_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["en", "cs", "fr", "de", "ar"])
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    if not os.path.isdir(f"figures/{args.dataset}/{args.model}"):
        os.makedirs(f"figures/{args.dataset}/{args.model}")
    word_dfs = {}
    print("hi")

    skip = []
    for l in args.langs:
        try:
            df_cap = pd.read_csv(f"outputs/{args.dataset}/pos/results_{l}.csv")
            df_txt = pd.read_csv(f"outputs/{args.dataset}/{args.model}/results_{l}.csv")
            assert len(df_cap) == len(df_txt)

            # remove lines where the token is <eos>
            df_cap = df_cap[df_cap.token != "<eos>"]
            # df_txt = df_txt[df_txt.token !=]
            # make index match
            # df_cap = df_cap.reset_index(drop=True)
            df = df_cap
            df[f"{args.model}_surprisal"] = df_txt[f"{args.model}_surprisal"]
            df["mutual_information"] = (
                df[f"{args.model}_surprisal"] - df_cap["paligemma_surprisal"]
            )
            # df = pd.read_csv(f"outputs/results_{l}_{args.dataset}_tagged.csv")
        except FileNotFoundError:
            skip.append(l)
            print(f"Error, {l} results not found for {args.dataset}", file=sys.stderr)
            continue
        grps = (
            df.word_stanza != df.word_stanza.shift()
        ).cumsum()  # TODO: fix for repeated words
        by_word = df.groupby(grps)[["mutual_information"]].transform("sum")
        df = df[["sentence", "POS", "word_stanza", "caption"]].join(by_word)
        df = df.groupby(grps).first().reset_index(drop=True)
        df = df.groupby("sentence").apply(lambda x: x.iloc[1:]).reset_index(drop=True)
        df = df[df.mutual_information != 0]
        # remove first word of each sentence
        word_dfs[l] = df
    pos_order = {}
    fig, axs = plt.subplots(4, 4, figsize=(40, 24), sharey=True)
    count = 0
    plt_num = 0
    for l in args.langs:
        # plot_mi(word_dfs[l], l, args.dataset)
        if l in skip:
            print(f"Skipping {l}")
            continue
        print(f"Plotting {l}")
        pos_order[l] = plot_pos(word_dfs[l], l, args.dataset, args.model)
        plot_pos(
            word_dfs[l], l, args.dataset, args.model, ax=axs[count // 4, count % 4]
        )
        count += 1
        if count % 16 == 0:
            fig.suptitle(
                f"Relative groundedness of POS in {args.dataset} dataset by language"
            )
            # for ax in axs.flat:
            #     ax.label_outer()
            fig.savefig(
                f"figures/{args.dataset}/{args.model}/pos_{(count-15)}-{count}.png",
                bbox_inches="tight",
            )
            fig, ax = plt.subplots(4, 4, figsize=(20, 20), sharey=True)
            plt_num += 1
    if count % 16 != 0:
        fig.suptitle(
            f"Relative groundedness of POS in {args.dataset} dataset by language"
        )
        # for ax in axs.flat:
        #     ax.label_outer()
        fig.savefig(
            f"figures/{args.dataset}/{args.model}/pos_{(count//16)*16+1}-{count}.png",
            bbox_inches="tight",
        )
    pos_values = pd.concat(pos_order.values()).reset_index()
    order = (
        pos_values.groupby("POS")
        .mean()
        .sort_values(by="mutual_information", ascending=True)
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # sns.boxplot(data=pos_values, x="POS", y="mutual_information", order=order.index)
    # print(mean)
    stds = pd.concat(pos_order.values(), axis=1).groupby(level=0, axis=1).std()
    # make order of std match order of means
    stds = stds.reindex(order.index)
    print(stds)
    ax.errorbar(
        order.index,
        order.mutual_information,
        yerr=stds.mutual_information,
        fmt="o",
        capsize=5,
        color="skyblue",
        markersize=8,
        elinewidth=2,
        markeredgecolor="blue",
        markerfacecolor="white",
    )
    print(order)
    print(stds)
    plt.axhline(y=0)
    ax.set_xlabel("Part of Speech")
    ax.set_ylabel("PMI")
    ax.set_title(f"PMI by Part of Speech, {args.dataset} dataset")
    plt.savefig(f"figures/{args.dataset}/{args.model}/pos_all.png", bbox_inches="tight")
    breakpoint()

    for pos1, pos2 in itertools.combinations(order, 2):
        t, p = ttest_ind(
            pos_values[pos_values.POS == pos1].mutual_information,
            pos_values[pos_values.POS == pos2].mutual_information,
        )
        print(f"{pos1} vs {pos2}: t={t}, p={p}")

    # add error bars
    # add error bars
    # tau_table = pd.DataFrame(index=args.langs, columns=args.langs)
    # rho_table = pd.DataFrame(index=args.langs, columns=args.langs)
    # tau_table = pd.DataFrame(columns=["lang1", "lang2", "tau"])
    # rho_table = pd.DataFrame(columns=["lang1", "lang2", "rho"])
    tau_table = []
    rho_table = []
    for l1, l2 in itertools.product(args.langs, args.langs):
        # deal with missing items in the correlation lists and turn into list
        order_l1 = list(pos_order[l1])
        order_l2 = list(pos_order[l2])
        missing = set(order_l1) - set(order_l2)
        for m in missing:
            order_l1.remove(m)
        missing = set(order_l2) - set(order_l1)
        for m in missing:
            order_l2.remove(m)
        if l1 != l2:
            print(order_l1, order_l2)
        print(l1, l2)
        print(order_l1, order_l2)
        tau = kendalltau(order_l1, order_l2).correlation
        rho = spearmanr(order_l1, order_l2).correlation
        print(tau, rho)
        tau_table.append({"lang1": l1, "lang2": l2, "tau": tau})
        rho_table.append({"lang1": l1, "lang2": l2, "rho": rho})
        # tau_table.at[l1, l2] = tau
        # rho_table.at[l1, l2] = rho
    tau_table = pd.DataFrame(tau_table)
    rho_table = pd.DataFrame(rho_table)
    print(tau_table)
    print(rho_table)
    pivot_tau = tau_table.pivot_table(index="lang1", columns="lang2", values="tau")
    pivot_rho = rho_table.pivot_table(index="lang1", columns="lang2", values="rho")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(pivot_tau, annot=False, cmap="RdBu", ax=ax, vmin=-1, vmax=1)
    plt.savefig(f"figures/tau_{args.dataset}.png", bbox_inches="tight")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(pivot_rho, annot=False, cmap="RdBu", ax=ax)

    plt.savefig(f"figures/rho_{args.dataset}.png", bbox_inches="tight", vmin=-1, vmax=1)


if __name__ == "__main__":
    main()
