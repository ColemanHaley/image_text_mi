import argparse
import itertools
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from iso639 import Lang
from scipy.stats import kendalltau, spearmanr, ttest_ind


plt.style.use("./opinionated_ch.mplstyle")
LANGS=['ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa',
  'fi', 'fil', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it',
  'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
  'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh'
] # Drop quz b/c of COCO.


def prepare_df(df_cap, df_txt, model):
    df_cap = df_cap[df_cap.token != "<eos>"]
    df = df_cap
    
    df[f"{model}_surprisal"] = df_txt[f"{model}_surprisal"]
    df["mutual_information"] = (
        df[f"{model}_surprisal"] - df_cap["paligemma_surprisal"]
    )
    grps = (
        df.word_stanza != df.word_stanza.shift()
    ).cumsum()  # TODO: fix for repeated words
    by_word = df.groupby(grps)[["mutual_information"]].transform("sum")
    df = df[["sentence", "POS", "word_stanza", "caption", "sentence_idx", "paligemma_surprisal", f"{model}_surprisal"]].join(by_word)
    df = df.groupby(grps).first().reset_index(drop=True)
    #df = df.groupby("sentence").apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    df = df[df.mutual_information != 0]
    return df


def load_dataset(dataset, model="ft-pali"):
    if not os.path.isdir(f"figures/{dataset}/{model}"):
        os.makedirs(f"figures/{dataset}/{model}")
    dfs = []
    for l in LANGS:
        try:
            df_cap = pd.read_csv(f"../outputs/{dataset}/pos/results_{l}.csv")
            df_txt = pd.read_csv(f"../outputs/{dataset}/{model}/results_{l}.csv")
            assert len(df_cap) == len(df_txt)
        except FileNotFoundError as e:
            print(e)
            print(f"Error, {l} results not found for {dataset}")
            continue
        df = prepare_df(df_cap, df_txt, model)
        df["lang"] = l
        print(df)
        dfs.append(df)
    return pd.concat(dfs)


#df_coco = load_dataset("coco35")
df_xm = load_dataset("xm3600")
df_multi = load_dataset("multi30k")
df_multitrain = load_dataset("multi30k_train")

df_xm.lang.unique()[0]

plt_df = df_xm[df_xm.sentence_idx < 20]
g = sns.FacetGrid(plt_df, col="lang", col_wrap=7, aspect=2)
g.map(sns.barplot, "sentence_idx", "mutual_information")

from wordfreq import zipf_frequency
df_xm = df_xm[df_xm.word_stanza != "#SKIP#"]


def get_freq(d):
    try:
        return zipf_frequency(d.word_stanza, d.lang, minimum=-1)
    except:
        return -1
df_xm["freq"] = df_xm.apply(lambda d: get_freq(d), axis=1)

(df_xm.lang == None).sum()

df

sns.scatterplot(df_xm, x="ft-pali_surprisal", y="mutual_information")

df_freq = df_type[df_type.freq > 0]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
X = df_freq.freq.to_numpy().reshape(-1, 1)
model.fit(X, df_freq["mutual_information"])
y_pred = model.predict(X)
print(f"R^2: {r2_score(df_freq['mutual_information'], y_pred)}")
sns.scatterplot(df_freq, x="freq", y="mutual_information")

# +

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
X = df_en.freq.to_numpy().reshape(-1, 1)
model.fit(X, df_en["mutual_information"])
y_pred = model.predict(X)
print(f"R^2: {r2_score(df_en['mutual_information'], y_pred)}")
sns.scatterplot(df_en, x="freq", y="mutual_information")
# -

df_type[df_type['word_stanza'] == 'green']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
X = df_en["IMAG.M"].to_numpy().reshape(-1, 1)
model.fit(X, df_en["mutual_information"])
y_pred = model.predict(X)
print(f"R^2: {r2_score(df_en['mutual_information'], y_pred)}")
sns.scatterplot(df_en, x="IMAG.M", y="mutual_information")

df_en

norms = pd.read_csv('~/Downloads/glasgownorms.csv', header=[0, 1])
cols = ['AROU', 'VAL', 'DOM', 'CNC', 'IMAG', 'FAM', 'AOA', 'SIZE', 'GEND']
stats = ['M', 'SD', 'N']
cols = ["Word", "Length"] + [f"{c}.{s}" for c in cols for s in stats]
norms.columns = cols

df_en[df_en.word_stanza == 'green']

df_en = df_type[df_type["fixi"] == 'en']

df_type

df_en = pd.merge(df_en, norms[['Word', 'CNC.M', 'IMAG.M']], left_on='word_stanza', right_on='Word', how='left')
df_en = df_en[~df_en["Word"].isna()]

df_en

sns.scatterplot(df_xm, x="paligemma_surprisal", y="mutual_information")

df_xm


def filter_infreq_types(df, threshold=50):
    df["fixit"] = df.word_stanza + "#" + df.POS + "#" + df.lang
    category_counts = df.fixit.value_counts()

    # Set a threshold for the minimum frequency required to keep a category
    # Filter categories based on frequency
    valid_categories = category_counts[category_counts >= threshold].index

    # Filter the DataFrame based on valid categories
    filtered = df[df.fixit.isin(valid_categories)]
    print(filtered)
    filtered = filtered.groupby("fixit").mean().reset_index()
    return filtered


df_type = filter_infreq_types(df_xm)

pos_means = {}
for l in df_xm.lang.unique():
    pos_means[l] = (
            df_xm[df_xm["lang"] == l].groupby([ "POS"])["mutual_information"].mean().sort_values(ascending=True)
    )

pos_means

df.lang

POS_FOR_ANALYSIS = [
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
from scipy.stats import weightedtau
def compare_langs(df):
    df = df[df.POS.isin(POS_FOR_ANALYSIS)]
    pos_means = {}
    for l in df.lang.unique():
        pos_means[l] = (
                df[df["lang"] == l].groupby([ "POS"])["mutual_information"].mean().sort_values(ascending=True)
        )
    tau_table = []
    rho_table = []
    for l1, l2 in itertools.product(df.lang.unique(), df.lang.unique()):
        # deal with missing items in the correlation lists and turn into list
        order_l1 = list(pos_means[l1].index)
        order_l2 = list(pos_means[l2].index)
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
    plt.savefig(f"figures/xm3600/tau.png", bbox_inches="tight")
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(pivot_rho, annot=False, cmap="RdBu", ax=ax)
    plt.show()

    plt.savefig(f"figures/xm3600/rho.png", bbox_inches="tight")
    


def plot_pos(df, lang, dataset, model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
        ax.set_title(
            f"Relative Groundedness of POS in {Lang(lang).name} for {dataset} dataset"
        )
    else:
        fig = None
        ax.set_title(f"{Lang(lang).name}")

    df_fig = df[df.POS.isin(POS_FOR_ANALYSIS)]
    # df_fig = df_fig.groupby(["word_stanza", "POS"]).mean().reset_index()
    group_means = (
        df_fig.groupby(["POS"])["mutual_information"].mean().sort_values(ascending=True)
    )
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
    plt.show()
    if fig is not None:
        fig.savefig(f"figures/{dataset}/{model}/pos_{lang}.png", bbox_inches="tight")
    return group_means


plot_pos(df_xm[df_xm.lang == "nl"], "nl", "xm3600", "ft-pali")

compare_langs(df_xm)

# +

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


# -

def plot_mi(df, lang, dataset):
    sns.displot(df, x="mutual_information", kind="kde")
    plt.title(f"{lang} {dataset} - Mutual Information")
    plt.xlabel("Mutual Information")
    plt.ylabel("Density")
    plt.savefig(f"figures/mi_{lang}_{dataset}.png", backend="pgf", bbox_inches="tight")


def plot_pos(df, lang, dataset, model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))
        ax.set_title(
            f"Relative Groundedness of POS in {Lang(lang).name} for {dataset} dataset"
        )
    else:
        fig = None
        ax.set_title(f"{Lang(lang).name}")



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
