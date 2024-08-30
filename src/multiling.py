import argparse
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, ttest_ind
import seaborn as sns


def plot_format():
    plt.rcParams["pgf.rcfonts"] = False
    plt.style.use(
        "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
    )
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams["pgf.preamble"] = "\n".join(
        [
            r"\usepackage[dvipsnames]{xcolor}",
            r"\usepackage{mathspec}",
            # r'\usepackage[no-math]{fontspec}',
            r"\setsansfont{EB Garamond}",
            r"\setallmainfonts{EB Garamond}",
            r"\color{white}",
            # r"\setmathsfont(Digits,Latin){Charis SIL Compact}",
        ]
    )


def plot_mi(df, lang, dataset):
    plot_format()
    sns.displot(df, x="mutual_information", kind="kde")
    plt.title(f"{lang} {dataset} - Mutual Information")
    plt.xlabel("Mutual Information")
    plt.ylabel("Density")
    plt.savefig(f"figures/mi_{lang}_{dataset}.png", backend="pgf", bbox_inches="tight")


def plot_pos(df, lang, dataset):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    df.fixit = df.word_stanza + "#" + df.POS
    category_counts = df.fixit.value_counts()
    print(f"LANG: {lang}")

    # Set a threshold for the minimum frequency required to keep a category
    threshold = 50

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
    # df_fig = df_fig.groupby(["word_stanza", "POS"]).mean().reset_index()
    group_means = (
        df_fig.groupby(["POS"])["mutual_information"].mean().sort_values(ascending=True)
    )
    print(group_means)
    sns.boxplot(
        data=df_fig, x="POS", y="mutual_information", order=group_means.index, ax=ax
    )
    plt.axhline(y=0)
    ax.tick_params(labelsize="large")
    ax.set_xlabel(
        "Part of Speech"
    )  # "$\mathrm{var}(\Delta_{\\mathrm{distribution}})$")
    ax.set_ylabel("Groundedness")
    plt.savefig(f"figures/pos_{lang}_{dataset}.png", backend="pgf", bbox_inches="tight")
    by_sent = df_fig.groupby("sentence").mean()
    # get mean value for sent
    print(by_sent["mutual_information"].mean(), by_sent["mutual_information"].std())
    return group_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    word_dfs = {}
    print("hi")
    for l in args.langs:
        df = pd.read_csv(f"outputs/results_{l}_{args.dataset}_tagged.csv")
        grps = (
            df.word_stanza != df.word_stanza.shift()
        ).cumsum()  # TODO: fix for repeated words
        by_word = df.groupby(grps)[
            ["txt_xent", "cap_xent", "mutual_information"]
        ].transform("sum")
        df = df[["sentence", "POS", "word_stanza", "caption"]].join(by_word)
        df = df.groupby(grps).first().reset_index(drop=True)
        word_dfs[l] = df
    pos_order = {}
    for l in args.langs:
        # plot_mi(word_dfs[l], l, args.dataset)
        pos_order[l] = plot_pos(word_dfs[l], l, args.dataset)
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
    plt.axhline(y=0)
    ax.set_xlabel("Part of Speech")
    ax.set_ylabel("PMI")
    ax.set_title(f"PMI by Part of Speech, {args.dataset} dataset")
    plt.savefig(
        f"figures/pos_all_{args.dataset}.png", backend="pgf", bbox_inches="tight"
    )

    # for pos1, pos2 in itertools.combinations(order, 2):
    #     t, p = ttest_ind(
    #         pos_values[pos_values.POS == pos1].mutual_information,
    #         pos_values[pos_values.POS == pos2].mutual_information,
    #     )
    #     print(f"{pos1} vs {pos2}: t={t}, p={p}")

    # add error bars
    # add error bars
    # tau_table = pd.DataFrame(index=args.langs, columns=args.lang)
    # rho_table = pd.DataFrame(index=args.langs, columns=args.langs)
    # tau_table = pd.DataFrame(columns=["lang1", "lang2", "tau"])
    # rho_table = pd.DataFrame(columns=["lang1", "lang2", "rho"])
    # for l1, l2 in itertools.product(args.langs, args.langs):
    #     # deal with missing items in the correlation lists and turn into list
    #     order_l1 = list(pos_order[l1])
    #     order_l2 = list(pos_order[l2])
    #     missing = set(order_l1) - set(order_l2)
    #     for m in missing:
    #         order_l1.remove(m)
    #     missing = set(order_l2) - set(order_l1)
    #     for m in missing:
    #         order_l2.remove(m)
    #     if l1 != l2:
    #         print(order_l1, order_l2)
    #     tau = kendalltau(order_l1, order_l2).correlation
    #     rho = spearmanr(order_l1, order_l2).correlation
    #     tau_table = tau_table.append(
    #         {"lang1": l1, "lang2": l2, "tau": tau}, ignore_index=True
    #     )
    #     rho_table = rho_table.append(
    #         {"lang1": l1, "lang2": l2, "rho": rho}, ignore_index=True
    #     )
    #     # tau_table.at[l1, l2] = tau
    #     # rho_table.at[l1, l2] = rho
    # print(tau_table)
    # print(rho_table)
    # pivot_tau = tau_table.pivot_table(index="lang1", columns="lang2", values="tau")
    # pivot_rho = rho_table.pivot_table(index="lang1", columns="lang2", values="rho")
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # sns.heatmap(pivot_tau, annot=False, cmap="RdBu", ax=ax, vmin=-1, vmax=1)
    # plt.savefig(f"figures/tau_{args.dataset}.png", bbox_inches="tight")
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # sns.heatmap(pivot_rho, annot=False, cmap="RdBu", ax=ax)

    # plt.savefig(f"figures/rho_{args.dataset}.png", bbox_inches="tight", vmin=-1, vmax=1)


if __name__ == "__main__":
    main()
