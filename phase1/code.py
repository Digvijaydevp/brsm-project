"""
Sentence Memorability — Analysis (Report Version)
Primary DVs: IR Accuracy (Hit Rate), WR Accuracy
Additional DV: Reaction Time (RT_IR)
Factors: Condition, Voice, Block, SubjectMem, ObjectMem

"""

import pandas as pd
import numpy as np
import glob
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings
from io import StringIO
from scipy import stats
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Output directories ──
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, 'analysis_results.txt')
log_buf = StringIO()

def log(msg=''):
    print(msg)
    log_buf.write(msg + '\n')

def save_results():
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write(log_buf.getvalue())
    log(f"\n[Saved text results to {RESULTS_FILE}]")

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log(f"  -> Saved figures/{name}")


# ════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & PREPARATION 
# ════════════════════════════════════════════════════════════════

def phase1_load():
    log("=" * 70)
    log("PHASE 1: DATA LOADING & PREPARATION")
    log("=" * 70)

    files = glob.glob(os.path.join(OUT_DIR, '*.log'))
    if not files:
        log("ERROR: No .log files found!"); sys.exit(1)
    log(f"  Found {len(files)} log files")

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype=str)
            frames.append(d)
        except Exception as e:
            log(f"  Warning: could not read {os.path.basename(f)}: {e}")
    df = pd.concat(frames, ignore_index=True)
    log(f"  Total rows loaded: {len(df)}")

    # Numeric conversions
    df['Accuracy_IR'] = pd.to_numeric(df['Accuracy IR'], errors='coerce')
    df['Accuracy_WR'] = pd.to_numeric(df['Accuracy WR'], errors='coerce')
    df['RT_IR'] = pd.to_numeric(df['Reaction_time_IR'], errors='coerce')
    df['RT_WR'] = pd.to_numeric(df['Reaction_time_WR'], errors='coerce')

    # Assign blocks per participant
    blocks_list = []
    for pid, grp in df.groupby('participant_ID', sort=False):
        block_num = 1
        bnums = []
        for ev in grp['Event']:
            bnums.append(block_num)
            if ev == 'Rest Phase started':
                block_num += 1
        g = grp.copy()
        g['Block'] = bnums
        blocks_list.append(g)
    df = pd.concat(blocks_list, ignore_index=True)

    # Exclude practice rows
    practice_mask = df['Event'].str.contains('Practice', na=False)
    n_practice = practice_mask.sum()
    df_main = df[~practice_mask].copy()
    log(f"  Excluded {n_practice} practice rows -> {len(df_main)} main rows")

    # Validation formula per (participant, block)
    def calc_validation(g):
        correct_val = (g['Event'] == 'Validation IR pressed').sum()
        wrong_ir = ((g['Event'] == 'Validation Wrong IR pressed') |
                    (g['Event'] == 'Wrong IR pressed')).sum()
        missed_val = (g['Event'] == 'Validation Missed').sum()
        passed = correct_val > (wrong_ir / 2.0) + missed_val
        return pd.Series({'CorrectVal': correct_val, 'WrongIR': wrong_ir,
                          'MissedVal': missed_val, 'Passed': passed})

    val_results = df_main.groupby(['participant_ID', 'Block']).apply(calc_validation).reset_index()
    n_total_blocks = len(val_results)
    valid_blocks = val_results[val_results['Passed'] == True]
    n_valid = len(valid_blocks)
    n_excluded = n_total_blocks - n_valid
    log(f"  Validation: {n_valid}/{n_total_blocks} blocks passed ({n_excluded} excluded)")

    df_valid = df_main.merge(valid_blocks[['participant_ID', 'Block']],
                             on=['participant_ID', 'Block'])
    log(f"  Rows after block exclusion: {len(df_valid)}")

    # Extract condition, voice, SubjectMem, ObjectMem from Stimulus
    def parse_stimulus(s):
        if pd.isna(s) or s == 'N/A':
            return pd.Series({'Condition': np.nan, 'Voice': np.nan,
                              'SubjectMem': np.nan, 'ObjectMem': np.nan})
        parts = str(s).split('_')
        cond = parts[0] if len(parts) >= 1 else np.nan
        voice = parts[-1] if len(parts) >= 3 else np.nan
        subj = obj = np.nan
        cond_map = {'HVL': 'HL', 'LVH': 'LH', 'LVL': 'LL'}
        cond = cond_map.get(cond, cond)
        if cond == 'HH':   subj, obj = 'H', 'H'
        elif cond == 'HL': subj, obj = 'H', 'L'
        elif cond == 'LH': subj, obj = 'L', 'H'
        elif cond == 'LL': subj, obj = 'L', 'L'
        return pd.Series({'Condition': cond, 'Voice': voice,
                          'SubjectMem': subj, 'ObjectMem': obj})

    stim_info = df_valid['Stimulus'].apply(parse_stimulus)
    df_valid = pd.concat([df_valid, stim_info], axis=1)

    # ── Build trial-level dataframe for target repeats ──
    presentations = df_valid[
        (df_valid['isTarget'] == 'true') &
        (df_valid['isRepeat'] == 'true') &
        (df_valid['Condition'].isin(['HH', 'HL', 'LH', 'LL'])) &
        (df_valid['Event'] == 'Sentence shown')
    ][['participant_ID', 'Stimulus', 'Block', 'Condition', 'Voice',
       'SubjectMem', 'ObjectMem']].copy()

    ir_pressed = df_valid[
        (df_valid['isTarget'] == 'true') &
        (df_valid['isRepeat'] == 'true') &
        (df_valid['Condition'].isin(['HH', 'HL', 'LH', 'LL'])) &
        (df_valid['Event'] == 'IR pressed')
    ][['participant_ID', 'Stimulus', 'Accuracy_IR', 'RT_IR']].copy()

    wr_rows = df_valid[
        (df_valid['Event'] == 'WR pressed') &
        (df_valid['isTarget'] == 'true') &
        (df_valid['isRepeat'] == 'true')
    ][['participant_ID', 'Stimulus', 'Accuracy_WR', 'RT_WR']].copy()

    targets_repeat = presentations.merge(ir_pressed, on=['participant_ID', 'Stimulus'], how='left')
    targets_repeat['IR_Hit'] = targets_repeat['Accuracy_IR'].notna().astype(int)
    targets_repeat['Accuracy_IR'] = targets_repeat['IR_Hit']
    targets_repeat = targets_repeat.merge(wr_rows, on=['participant_ID', 'Stimulus'], how='left')

    hits = targets_repeat[targets_repeat['IR_Hit'] == 1].copy()

    n_presentations = len(presentations)
    n_hits = len(hits)
    n_misses = n_presentations - n_hits
    n_with_wr = hits['Accuracy_WR'].notna().sum()
    hit_rate = n_hits / n_presentations if n_presentations > 0 else 0

    log(f"\n  ── Trial-Level Data (target repeats) ──")
    log(f"  Sentence presentations: {n_presentations}")
    log(f"  Hits (IR pressed): {n_hits}")
    log(f"  Misses (no response): {n_misses}")
    log(f"  Overall Hit Rate: {hit_rate:.4f}")
    log(f"  Hits with WR data: {n_with_wr} ({100*n_with_wr/n_hits:.1f}%)")

    # False alarms
    false_alarms = df_valid[
        df_valid['Event'].isin(['Validation Wrong IR pressed', 'Wrong IR pressed'])
    ].copy()

    # Per-participant False Alarm Rate
    lure_shown = df_valid[
        (df_valid['Event'] == 'Sentence shown') &
        ~((df_valid['isTarget'] == 'true') & (df_valid['isRepeat'] == 'true'))
    ]
    fa_per_ppt = false_alarms.groupby('participant_ID').size().reset_index(name='n_fa')
    lure_per_ppt = lure_shown.groupby('participant_ID').size().reset_index(name='n_lure')
    fa_rate_df = lure_per_ppt.merge(fa_per_ppt, on='participant_ID', how='left')
    fa_rate_df['n_fa'] = fa_rate_df['n_fa'].fillna(0).astype(int)
    fa_rate_df['FA_Rate'] = fa_rate_df['n_fa'] / fa_rate_df['n_lure']

    targets_repeat = targets_repeat.merge(fa_rate_df[['participant_ID', 'FA_Rate']],
                                          on='participant_ID', how='left')
    targets_repeat['FA_Rate'] = targets_repeat['FA_Rate'].fillna(0)

    log(f"\n  ── False Alarm Rate ──")
    log(f"  Total false alarm events: {len(false_alarms)}")
    log(f"  Total lure presentations: {len(lure_shown)}")
    log(f"  Overall FA Rate: {false_alarms.shape[0] / lure_shown.shape[0]:.4f}")
    log(f"  Per-participant FA Rate: mean={fa_rate_df['FA_Rate'].mean():.4f}, "
        f"median={fa_rate_df['FA_Rate'].median():.4f}, "
        f"SD={fa_rate_df['FA_Rate'].std():.4f}")
    log(f"  Range: {fa_rate_df['FA_Rate'].min():.4f} to {fa_rate_df['FA_Rate'].max():.4f}")

    hits = targets_repeat[targets_repeat['IR_Hit'] == 1].copy()

    n_participants = df_valid['participant_ID'].nunique()
    log(f"\n  Unique participants in valid data: {n_participants}")
    log(f"  Target repeat trials: {n_presentations}")
    log(f"  Hits (IR correct, for RT & WR): {n_hits}")
    log(f"  False alarm events: {len(false_alarms)}")

    # ── Per-sentence IR Memorability Score ──
    per_sent_shown = presentations.groupby('Stimulus').size().reset_index(name='n_shown')
    per_sent_hits = ir_pressed.groupby('Stimulus').size().reset_index(name='n_hits')
    per_sentence_ir = per_sent_shown.merge(per_sent_hits, on='Stimulus', how='left')
    per_sentence_ir['n_hits'] = per_sentence_ir['n_hits'].fillna(0).astype(int)
    per_sentence_ir['IR_Memorability'] = per_sentence_ir['n_hits'] / per_sentence_ir['n_shown']
    per_sentence_ir['Condition'] = per_sentence_ir['Stimulus'].str.extract(r'^(HH|HVL|LVH|LVL)')
    per_sentence_ir['Condition'] = per_sentence_ir['Condition'].replace({'HVL': 'HL', 'LVH': 'LH', 'LVL': 'LL'})
    per_sentence_ir['Voice'] = per_sentence_ir['Stimulus'].str.extract(r'_(A|P)$')

    log(f"\n  ── Per-Sentence IR Memorability Scores ──")
    log(f"  Unique sentences: {len(per_sentence_ir)}")
    log(f"  Mean IR Memorability: {per_sentence_ir['IR_Memorability'].mean():.4f}")
    log(f"  Median: {per_sentence_ir['IR_Memorability'].median():.4f}")
    log(f"  SD: {per_sentence_ir['IR_Memorability'].std():.4f}")
    log(f"  Range: {per_sentence_ir['IR_Memorability'].min():.4f} to {per_sentence_ir['IR_Memorability'].max():.4f}")

    # ── Per-sentence WR Accuracy ──
    wr_per_sent = wr_rows.dropna(subset=['Accuracy_WR']).copy()
    if len(wr_per_sent) > 0:
        per_sent_wr_n = wr_per_sent.groupby('Stimulus').size().reset_index(name='n_wr')
        per_sent_wr_acc = wr_per_sent.groupby('Stimulus')['Accuracy_WR'].mean().reset_index()
        per_sent_wr_acc.columns = ['Stimulus', 'WR_Memorability']
        per_sentence_wr = per_sent_wr_n.merge(per_sent_wr_acc, on='Stimulus')
        per_sentence_wr['Condition'] = per_sentence_wr['Stimulus'].str.extract(r'^(HH|HVL|LVH|LVL)')
        per_sentence_wr['Condition'] = per_sentence_wr['Condition'].replace({'HVL': 'HL', 'LVH': 'LH', 'LVL': 'LL'})
        per_sentence_wr['Voice'] = per_sentence_wr['Stimulus'].str.extract(r'_(A|P)$')
        log(f"\n  ── Per-Sentence WR Accuracy ──")
        log(f"  Unique sentences with WR: {len(per_sentence_wr)}")
        log(f"  Mean WR Accuracy: {per_sentence_wr['WR_Memorability'].mean():.4f}")
    else:
        per_sentence_wr = pd.DataFrame()

    return (df_valid, val_results, targets_repeat, hits,
            false_alarms, per_sentence_ir, per_sentence_wr, fa_rate_df)


# ════════════════════════════════════════════════════════════════
# PHASE 2: DESCRIPTIVE STATISTICS & VISUALIZATIONS  
# ════════════════════════════════════════════════════════════════

def phase2_descriptive(targets_repeat, hits, per_sentence_ir, per_sentence_wr, fa_rate_df):
    log("\n" + "=" * 70)
    log("PHASE 2: DESCRIPTIVE STATISTICS & VISUALIZATIONS")
    log("=" * 70)

    conds = ['HH', 'HL', 'LH', 'LL']
    wr_hits = hits.dropna(subset=['Accuracy_WR'])

    # ────────────────────────────────────────────────────────────
    # 2A. Per-Sentence Memorability Distributions
    # ────────────────────────────────────────────────────────────
    log("\n── 2A. Per-Sentence IR Memorability Score ──")
    ir_mem = per_sentence_ir['IR_Memorability']
    log(f"  N sentences: {len(per_sentence_ir)}")
    log(f"  Mean: {ir_mem.mean():.4f}")
    log(f"  Median: {ir_mem.median():.4f}")
    log(f"  Std Dev: {ir_mem.std():.4f}")
    log(f"  Skewness: {ir_mem.skew():.4f}")
    log(f"  Kurtosis: {ir_mem.kurtosis():.4f}")
    log(f"  Min: {ir_mem.min():.4f},  Max: {ir_mem.max():.4f}")

    log("\nBy Condition:")
    t = per_sentence_ir.groupby('Condition')['IR_Memorability'].agg(['count', 'mean', 'median', 'std']).round(4)
    t.columns = ['N_sentences', 'Mean', 'Median', 'SD']
    log(t.to_string())

    if len(per_sentence_wr) > 0:
        log("\n── 2A. Per-Sentence WR Accuracy ──")
        wr_mem = per_sentence_wr['WR_Memorability']
        log(f"  N sentences: {len(per_sentence_wr)}")
        log(f"  Mean: {wr_mem.mean():.4f}")
        log(f"  Median: {wr_mem.median():.4f}")
        log(f"  Std Dev: {wr_mem.std():.4f}")
        log(f"  Skewness: {wr_mem.skew():.4f}")
        log(f"  Kurtosis: {wr_mem.kurtosis():.4f}")
        log(f"  Min: {wr_mem.min():.4f},  Max: {wr_mem.max():.4f}")

    # ────────────────────────────────────────────────────────────
    # 2B. Overall Summary (trial-level)
    # ────────────────────────────────────────────────────────────
    log("\n── 2B. Overall Summary ──")
    log(f"  N trials: {len(targets_repeat)}")
    log(f"  Mean Hit Rate: {targets_repeat['Accuracy_IR'].mean():.4f}")
    log(f"  SD: {targets_repeat['Accuracy_IR'].std():.4f}")

    # ────────────────────────────────────────────────────────────
    # 2C. IR Accuracy by Condition (per-participant aggregated)
    # ────────────────────────────────────────────────────────────
    ppc_ir = targets_repeat.groupby(['participant_ID', 'Condition'])['Accuracy_IR'].mean().reset_index()
    ppc_ir.columns = ['participant_ID', 'Condition', 'HitRate']
    log("\n── 2C. IR Accuracy (Hit Rate) by Condition ──")
    t = ppc_ir.groupby('Condition')['HitRate'].agg(['count', 'mean', 'median', 'std']).round(4)
    t.columns = ['N_ppts', 'MeanHitRate', 'MedianHitRate', 'SD']
    log(t.to_string())

    # 95% CIs for Hit Rate
    ci_hr = {}
    log("\n── 95% CI for Hit Rate by Condition ──")
    log(f"  {'Condition':<10} {'N':>5} {'Mean':>10} {'SE':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}")
    log("  " + "-" * 65)
    for c in conds:
        vals = ppc_ir[ppc_ir['Condition'] == c]['HitRate'].dropna().values
        if len(vals) >= 2:
            n = len(vals)
            mean = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(n)
            t_crit = t_dist.ppf(0.975, df=n - 1)
            ci_lo = mean - t_crit * se
            ci_hi = mean + t_crit * se
            ci_hr[c] = (mean, ci_lo, ci_hi, se)
            log(f"  {c:<10} {n:>5} {mean:>10.4f} {se:>10.4f} {ci_lo:>14.4f} {ci_hi:>14.4f}")

    # ────────────────────────────────────────────────────────────
    # 2D. Corrected IR by Condition (Hit Rate − FA Rate)
    # ────────────────────────────────────────────────────────────
    ppc_corrected_ir = ppc_ir.merge(fa_rate_df[['participant_ID', 'FA_Rate']], on='participant_ID', how='left')
    ppc_corrected_ir['FA_Rate'] = ppc_corrected_ir['FA_Rate'].fillna(0)
    ppc_corrected_ir['CorrectedIR'] = ppc_corrected_ir['HitRate'] - ppc_corrected_ir['FA_Rate']
    log("\n── 2D. Corrected IR by Condition ──")
    t = ppc_corrected_ir.groupby('Condition')['CorrectedIR'].agg(['count', 'mean', 'median', 'std']).round(4)
    t.columns = ['N_ppts', 'MeanCorrIR', 'MedianCorrIR', 'SD']
    log(t.to_string())

    # ────────────────────────────────────────────────────────────
    # 2E. IR Accuracy by Voice
    # ────────────────────────────────────────────────────────────
    ppv_ir = targets_repeat.groupby(['participant_ID', 'Voice'])['Accuracy_IR'].mean().reset_index()
    ppv_ir.columns = ['participant_ID', 'Voice', 'HitRate']
    log("\n── 2E. IR Accuracy by Voice ──")
    t = ppv_ir.groupby('Voice')['HitRate'].agg(['count', 'mean', 'std']).round(4)
    log(t.to_string())

    # ────────────────────────────────────────────────────────────
    # 2F. IR Accuracy by Block
    # ────────────────────────────────────────────────────────────
    ppb_ir = targets_repeat.groupby(['participant_ID', 'Block'])['Accuracy_IR'].mean().reset_index()
    ppb_ir.columns = ['participant_ID', 'Block', 'HitRate']
    log("\n── 2F. IR Accuracy by Block ──")
    t = ppb_ir.groupby('Block')['HitRate'].agg(['count', 'mean', 'std']).round(4)
    log(t.to_string())

    # ────────────────────────────────────────────────────────────
    # 2G. WR Accuracy by Condition
    # ────────────────────────────────────────────────────────────
    ppc_wr = wr_hits.groupby(['participant_ID', 'Condition'])['Accuracy_WR'].mean().reset_index()
    ppc_wr.columns = ['participant_ID', 'Condition', 'WR_Rate']
    log("\n── 2G. WR Accuracy by Condition ──")
    if len(ppc_wr) > 0:
        t = ppc_wr.groupby('Condition')['WR_Rate'].agg(['count', 'mean', 'median', 'std']).round(4)
        t.columns = ['N_ppts', 'MeanWR', 'MedianWR', 'SD']
        log(t.to_string())

    # 95% CIs for WR Accuracy
    ci_wr = {}
    if len(ppc_wr) > 0:
        log("\n── 95% CI for WR Accuracy by Condition ──")
        for c in conds:
            vals = ppc_wr[ppc_wr['Condition'] == c]['WR_Rate'].dropna().values
            if len(vals) >= 2:
                n = len(vals)
                mean = vals.mean()
                se = vals.std(ddof=1) / np.sqrt(n)
                t_crit = t_dist.ppf(0.975, df=n - 1)
                ci_wr[c] = (mean, mean - t_crit * se, mean + t_crit * se, se)
                log(f"  {c}: Mean={mean:.4f}, 95% CI [{mean - t_crit * se:.4f}, {mean + t_crit * se:.4f}]")

    # WR by Block
    ppb_wr = wr_hits.groupby(['participant_ID', 'Block'])['Accuracy_WR'].mean().reset_index()
    ppb_wr.columns = ['participant_ID', 'Block', 'WR_Rate']

    # WR by Voice
    ppv_wr = wr_hits.groupby(['participant_ID', 'Voice'])['Accuracy_WR'].mean().reset_index()
    ppv_wr.columns = ['participant_ID', 'Voice', 'WR_Rate']

    # ────────────────────────────────────────────────────────────
    # 2H. 2×2 Factorial: SubjectMem × ObjectMem
    # ────────────────────────────────────────────────────────────
    pps = targets_repeat.groupby(['participant_ID', 'SubjectMem', 'ObjectMem'])['Accuracy_IR'].mean().reset_index()
    pps.columns = ['participant_ID', 'SubjectMem', 'ObjectMem', 'HitRate']
    log("\n── 2H. 2×2 Factorial: SubjectMem × ObjectMem (Hit Rate) ──")
    t = pps.groupby(['SubjectMem', 'ObjectMem'])['HitRate'].agg(['count', 'mean', 'std']).round(4)
    log(t.to_string())
    log("\n  Marginal by SubjectMem:")
    log(pps.groupby('SubjectMem')['HitRate'].mean().round(4).to_string())
    log("\n  Marginal by ObjectMem:")
    log(pps.groupby('ObjectMem')['HitRate'].mean().round(4).to_string())

    # ────────────────────────────────────────────────────────────
    # 2I. RT Descriptives
    # ────────────────────────────────────────────────────────────
    log("\n── 2I. Reaction Time IR (hits only) ──")
    rt = hits['RT_IR'].dropna()
    log(f"  N: {len(rt)}, Mean: {rt.mean():.2f}, Median: {rt.median():.2f}, SD: {rt.std():.2f}")

    ppc_rt = hits.groupby(['participant_ID', 'Condition'])['RT_IR'].median().reset_index()
    ppc_rt.columns = ['participant_ID', 'Condition', 'MedianRT']

    # ────────────────────────────────────────────────────────────
    # 2J. Correlation: IR vs WR (participant-level)
    # ────────────────────────────────────────────────────────────
    ppt_agg = targets_repeat.groupby('participant_ID').agg(
        HitRate=('Accuracy_IR', 'mean'),
        MeanRT_IR=('RT_IR', 'mean'),
    ).reset_index()
    wr_agg = wr_hits.groupby('participant_ID')['Accuracy_WR'].mean().reset_index()
    wr_agg.columns = ['participant_ID', 'WR_Accuracy']
    rt_wr_agg = hits.dropna(subset=['RT_WR']).groupby('participant_ID')['RT_WR'].mean().reset_index()
    rt_wr_agg.columns = ['participant_ID', 'MeanRT_WR']
    ppt_corr = ppt_agg.merge(wr_agg, on='participant_ID', how='left')
    ppt_corr = ppt_corr.merge(rt_wr_agg, on='participant_ID', how='left')

    log("\n── 2J. Pearson Correlations ──")
    for v1, v2 in [('HitRate', 'WR_Accuracy'), ('HitRate', 'MeanRT_IR')]:
        if v1 in ppt_corr.columns and v2 in ppt_corr.columns:
            valid = ppt_corr[[v1, v2]].dropna()
            if len(valid) >= 3:
                r, p = stats.pearsonr(valid[v1], valid[v2])
                log(f"  {v1} vs {v2}: r = {r:.4f}, p = {p:.4e} (N={len(valid)})")

    # ────────────────────────────────────────────────────────────
    # 2K. Assumption Checks
    # ────────────────────────────────────────────────────────────
    log("\n── 2K. Assumption Checking ──")
    log("  |skewness| < 2 and |kurtosis| < 7 → approximate normality")

    log(f"\n  {'DV':<16} {'Cond':<6} {'N':>5} {'Skew':>8} {'Kurt':>8} {'~Normal?':>10}")
    log("  " + "-" * 60)
    for c in conds:
        vals = ppc_ir[ppc_ir['Condition'] == c]['HitRate']
        if len(vals) >= 3:
            sk, ku = vals.skew(), vals.kurtosis()
            ok = "Yes" if abs(sk) < 2 and abs(ku) < 7 else "No"
            log(f"  {'Hit Rate':<16} {c:<6} {len(vals):>5} {sk:>8.4f} {ku:>8.4f} {ok:>10}")
    for c in conds:
        vals = ppc_wr[ppc_wr['Condition'] == c]['WR_Rate'] if len(ppc_wr) > 0 else pd.Series(dtype=float)
        if len(vals) >= 3:
            sk, ku = vals.skew(), vals.kurtosis()
            ok = "Yes" if abs(sk) < 2 and abs(ku) < 7 else "No"
            log(f"  {'WR Accuracy':<16} {c:<6} {len(vals):>5} {sk:>8.4f} {ku:>8.4f} {ok:>10}")

    # Variance homogeneity
    log("\n  Variance ratios (max/min < 4 → reasonable homogeneity):")
    for dv_name, df_chk, col in [('Hit Rate', ppc_ir, 'HitRate'), ('WR Accuracy', ppc_wr, 'WR_Rate')]:
        if len(df_chk) == 0:
            continue
        groups = [df_chk[df_chk['Condition'] == c][col].dropna().values for c in conds]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) == 4:
            variances = [np.var(g, ddof=1) for g in groups]
            ratio = max(variances) / min(variances) if min(variances) > 0 else np.inf
            log(f"  {dv_name}: ratio = {ratio:.2f} {'-> OK' if ratio < 4 else '-> Potential heterogeneity'}")

    # ════════════════════════════════════════════════════════════
    # VISUALIZATIONS — 11 report plots
    # ════════════════════════════════════════════════════════════
    log("\n── Generating 11 Report Plots ──")

    palette = {'HH': '#2ecc71', 'HL': '#3498db', 'LH': '#e67e22', 'LL': '#e74c3c'}
    voice_pal = {'A': '#3498db', 'P': '#e74c3c'}
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Plot 04: Violin IR Accuracy by Condition ──
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=ppc_ir, x='Condition', y='HitRate', order=conds, palette=palette,
                   inner='quartile', cut=0, ax=ax)
    ax.set_title('IR Accuracy by Condition — Violin', fontweight='bold')
    ax.set_ylabel('IR Accuracy')
    ax.set_ylim(-0.05, 1.15)
    save_fig(fig, 'Plot04_IR_ACC_Violin.png')

    # ── Plot 06: Violin IR Accuracy by Voice ──
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=ppv_ir, x='Voice', y='HitRate', order=['A', 'P'],
                   palette=voice_pal, inner='quartile', cut=0, ax=ax)
    ax.set_title('IR Accuracy by Voice — Violin', fontweight='bold')
    ax.set_ylabel('IR Accuracy')
    ax.set_xticklabels(['Active', 'Passive'])
    ax.set_ylim(-0.05, 1.15)
    save_fig(fig, 'Plot06_IR_ACC_Violin_Voice.png')

    # ── Plot 07: Boxplot IR Accuracy by Block ──
    blocks_sorted = sorted(ppb_ir['Block'].unique())
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=ppb_ir, x='Block', y='HitRate', order=blocks_sorted,
                color='#3498db', showfliers=True,
                flierprops=dict(marker='o', alpha=0.3, markersize=3), ax=ax)
    ax.set_title('IR Accuracy by Block — Boxplot', fontweight='bold')
    ax.set_ylabel('IR Accuracy')
    ax.set_ylim(0, 1.1)
    save_fig(fig, 'Plot07_IR_ACC_Boxplot_Block.png')

    # ── Plot 12: 2×2 Heatmap ──
    fig, ax = plt.subplots(figsize=(7, 5))
    heat_data = pps.groupby(['SubjectMem', 'ObjectMem'])['HitRate'].mean().unstack()
    sns.heatmap(heat_data, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
                linewidths=2, linecolor='black', cbar_kws={'label': 'Mean Hit Rate'})
    ax.set_title('2×2 Mean Hit Rate (SubjectMem × ObjectMem)', fontweight='bold')
    ax.set_ylabel('Subject Memorability')
    ax.set_xlabel('Object Memorability')
    save_fig(fig, 'Plot12_Heatmap_2x2.png')

    # ── Plot 16: Violin WR Accuracy by Condition ──
    if len(ppc_wr) > 0:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.violinplot(data=ppc_wr, x='Condition', y='WR_Rate', order=conds, palette=palette,
                       inner='quartile', cut=0, ax=ax)
        ax.set_title('WR Accuracy by Condition — Violin', fontweight='bold')
        ax.set_ylabel('WR Accuracy')
        ax.set_ylim(-0.05, 1.15)
        save_fig(fig, 'Plot16_WR_Accuracy_Violin.png')

    # ── Plot 21: IR vs WR Scatter ──
    fig, ax = plt.subplots(figsize=(8, 7))
    if 'WR_Accuracy' in ppt_corr.columns and ppt_corr['WR_Accuracy'].notna().sum() > 5:
        valid = ppt_corr[['HitRate', 'WR_Accuracy']].dropna()
        ax.scatter(valid['HitRate'], valid['WR_Accuracy'], alpha=0.5, s=40, color='#2c3e50')
        if len(valid) > 2 and valid['HitRate'].std() > 0:
            slope, intercept, r, p, se = stats.linregress(valid['HitRate'], valid['WR_Accuracy'])
            x_line = np.linspace(valid['HitRate'].min(), valid['HitRate'].max(), 100)
            ax.plot(x_line, intercept + slope * x_line, 'r-', lw=2,
                    label=f'r={r:.3f}, p={p:.3e}')
            ax.legend(fontsize=11)
        ax.set_xlabel('IR Accuracy (Hit Rate)')
        ax.set_ylabel('WR Accuracy')
        ax.set_title('IR vs WR Accuracy (per participant)', fontweight='bold')
    save_fig(fig, 'Plot21_IR_vs_WR_Scatter.png')

    # ── Plot 23: Histogram IR Memorability (per sentence) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(per_sentence_ir['IR_Memorability'], bins=30, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(per_sentence_ir['IR_Memorability'].mean(), color='red', ls='--', lw=2,
               label=f"Mean = {per_sentence_ir['IR_Memorability'].mean():.4f}")
    ax.axvline(per_sentence_ir['IR_Memorability'].median(), color='orange', ls='--', lw=2,
               label=f"Median = {per_sentence_ir['IR_Memorability'].median():.4f}")
    ax.set_xlabel('IR Memorability Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of IR Memorability Scores (per sentence)', fontweight='bold')
    ax.legend()
    save_fig(fig, 'Plot23_IR_Memorability_Histogram.png')

    # ── Plot 24: Histogram WR Accuracy (per sentence) ──
    if len(per_sentence_wr) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(per_sentence_wr['WR_Memorability'], bins=30, color='#8e44ad', edgecolor='white', alpha=0.8)
        ax.axvline(per_sentence_wr['WR_Memorability'].mean(), color='red', ls='--', lw=2,
                   label=f"Mean = {per_sentence_wr['WR_Memorability'].mean():.4f}")
        ax.axvline(per_sentence_wr['WR_Memorability'].median(), color='orange', ls='--', lw=2,
                   label=f"Median = {per_sentence_wr['WR_Memorability'].median():.4f}")
        ax.set_xlabel('WR Accuracy Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of WR Accuracy Scores (per sentence)', fontweight='bold')
        ax.legend()
        save_fig(fig, 'Plot24_WR_Accuracy_Histogram.png')

    # ── Plot 31: Boxplot Corrected IR by Condition ──
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=ppc_corrected_ir, x='Condition', y='CorrectedIR', order=conds,
                palette=palette, showfliers=True,
                flierprops=dict(marker='o', alpha=0.4, markersize=4), ax=ax)
    ax.set_title('Corrected IR by Condition — Boxplot', fontweight='bold')
    ax.set_ylabel('Corrected IR (Hit Rate − FA Rate)')
    save_fig(fig, 'Plot31_CorrectedIR_Boxplot.png')

    # ── Plot 35: Block Trend (3-panel) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    blk_means = ppb_ir.groupby('Block')['HitRate'].agg(['mean', 'std'])
    n_ppts = ppb_ir['participant_ID'].nunique()
    ax.errorbar(blk_means.index, blk_means['mean'], yerr=blk_means['std'] / np.sqrt(n_ppts),
                marker='o', capsize=5, linewidth=2, markersize=8, color='#2c3e50')
    ax.set_xlabel('Block'); ax.set_ylabel('Mean Hit Rate')
    ax.set_title('Hit Rate Across Blocks', fontweight='bold')

    ax = axes[1]
    if len(ppb_wr) > 0:
        blk_wr_means = ppb_wr.groupby('Block')['WR_Rate'].agg(['mean', 'std'])
        n_ppts_wr = ppb_wr['participant_ID'].nunique()
        ax.errorbar(blk_wr_means.index, blk_wr_means['mean'],
                    yerr=blk_wr_means['std'] / np.sqrt(n_ppts_wr),
                    marker='^', capsize=5, linewidth=2, markersize=8, color='#8e44ad')
    ax.set_xlabel('Block'); ax.set_ylabel('Mean WR Accuracy')
    ax.set_title('WR Accuracy Across Blocks', fontweight='bold')

    ax = axes[2]
    ppb_rt_plot = hits.groupby(['participant_ID', 'Block'])['RT_IR'].median().reset_index()
    blk_rt_means = ppb_rt_plot.groupby('Block')['RT_IR'].agg(['mean', 'std'])
    ax.errorbar(blk_rt_means.index, blk_rt_means['mean'],
                yerr=blk_rt_means['std'] / np.sqrt(ppb_rt_plot['participant_ID'].nunique()),
                marker='s', capsize=5, linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Block'); ax.set_ylabel('Median RT_IR (ms)')
    ax.set_title('RT Across Blocks', fontweight='bold')

    fig.suptitle('Performance Trends Across Blocks', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, 'Plot35_Block_Trend.png')

    # ── Plot 36: Correlation Heatmap ──
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_cols = ['HitRate', 'WR_Accuracy', 'MeanRT_IR', 'MeanRT_WR']
    existing_cols = [c for c in corr_cols if c in ppt_corr.columns and ppt_corr[c].notna().sum() > 5]
    if len(existing_cols) >= 2:
        corr_matrix = ppt_corr[existing_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    ax=ax, linewidths=1, square=True, vmin=-1, vmax=1)
        ax.set_title('Participant-Level Correlations (All DVs)', fontweight='bold')
    save_fig(fig, 'Plot36_Correlation_Heatmap.png')

    return (ppc_ir, ppc_rt, ppc_wr, ppv_ir, ppb_ir, ppv_wr, ppb_wr,
            pps, ppc_corrected_ir, ppt_corr)


# ════════════════════════════════════════════════════════════════
# PHASE 3: INFERENTIAL STATISTICS  
# ════════════════════════════════════════════════════════════════

def phase3_inferential(targets_repeat, hits, ppc_ir, ppc_rt, ppc_wr,
                       ppv_ir, ppb_ir, fa_rate_df, val_results, df_valid):
    log("\n" + "=" * 70)
    log("PHASE 3: INFERENTIAL STATISTICS")
    log("=" * 70)

    conds = ['HH', 'HL', 'LH', 'LL']
    wr_hits = hits.dropna(subset=['Accuracy_WR'])

    def paired_t_report(label, df_wide, col_a, col_b):
        if col_a in df_wide.columns and col_b in df_wide.columns:
            valid = df_wide[[col_a, col_b]].dropna()
            if len(valid) >= 2:
                t_stat, p = stats.ttest_rel(valid[col_a], valid[col_b])
                n = len(valid)
                d = t_stat / np.sqrt(n)
                log(f"  {label}: t({n-1}) = {t_stat:.4f}, p = {p:.4e}, Cohen's d = {d:.4f}")
                log(f"    {'*** SIGNIFICANT ***' if p < 0.05 else 'Not significant'}")
                return t_stat, p, d, n
        return None

    # ────────────────────────────────────────────────────────────
    # 3A. Kruskal-Wallis H Tests (4 conditions)
    # ────────────────────────────────────────────────────────────
    log("\n── 3A. Kruskal-Wallis H Tests (4 conditions) ──")

    for dv_name, dv_df, dv_col in [('Hit Rate', ppc_ir, 'HitRate'),
                                    ('WR Accuracy', ppc_wr, 'WR_Rate')]:
        groups = [dv_df[dv_df['Condition'] == c][dv_col].dropna().values for c in conds]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            k = len(groups)
            eta_sq = (h - k + 1) / (n_total - k) if n_total > k else np.nan
            log(f"  {dv_name:<16} H({k-1}) = {h:.4f}, p = {p:.4e}, eta2 = {eta_sq:.4f}")
            log(f"  {'':16} {'*** SIGNIFICANT ***' if p < 0.05 else 'Not significant'}")

    # KW across blocks
    log("\n  KW across Blocks (Hit Rate):")
    g_blk = [ppb_ir[ppb_ir['Block'] == b]['HitRate'].values for b in sorted(ppb_ir['Block'].unique())]
    g_blk = [g for g in g_blk if len(g) > 0]
    if len(g_blk) >= 2:
        h, p = stats.kruskal(*g_blk)
        log(f"  Hit Rate by Block: H = {h:.4f}, p = {p:.4e}")

    # KW for RT
    log("\n  KW — RT_IR by Condition:")
    groups = [ppc_rt[ppc_rt['Condition'] == c]['MedianRT'].dropna().values for c in conds]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        eta_sq = (h - k + 1) / (n_total - k) if n_total > k else np.nan
        log(f"  RT_IR            H({k-1}) = {h:.4f}, p = {p:.4e}, eta2 = {eta_sq:.4f}")

    # ────────────────────────────────────────────────────────────
    # 3B. Chi-Square Tests
    # ────────────────────────────────────────────────────────────
    log("\n── 3B. Chi-Square Tests ──")

    ct_ir_cond = pd.crosstab(targets_repeat['Condition'], targets_repeat['Accuracy_IR'])
    if ct_ir_cond.shape[1] == 2:
        chi2, p_chi, dof, _ = stats.chi2_contingency(ct_ir_cond)
        log(f"  IR by Condition: chi2({dof}) = {chi2:.4f}, p = {p_chi:.4e} {'***' if p_chi < 0.05 else ''}")

    ct_ir_voice = pd.crosstab(targets_repeat['Voice'], targets_repeat['Accuracy_IR'])
    if ct_ir_voice.shape[1] == 2:
        chi2, p_chi, dof, _ = stats.chi2_contingency(ct_ir_voice)
        log(f"  IR by Voice:     chi2({dof}) = {chi2:.4f}, p = {p_chi:.4e} {'***' if p_chi < 0.05 else ''}")

    if len(wr_hits) > 0:
        ct_wr_cond = pd.crosstab(wr_hits['Condition'], wr_hits['Accuracy_WR'])
        if ct_wr_cond.shape[1] >= 2:
            chi2, p_chi, dof, _ = stats.chi2_contingency(ct_wr_cond)
            log(f"  WR by Condition: chi2({dof}) = {chi2:.4f}, p = {p_chi:.4e} {'***' if p_chi < 0.05 else ''}")

        ct_wr_voice = pd.crosstab(wr_hits['Voice'], wr_hits['Accuracy_WR'])
        if ct_wr_voice.shape[1] >= 2:
            chi2, p_chi, dof, _ = stats.chi2_contingency(ct_wr_voice)
            log(f"  WR by Voice:     chi2({dof}) = {chi2:.4f}, p = {p_chi:.4e} {'***' if p_chi < 0.05 else ''}")

    # ────────────────────────────────────────────────────────────
    # 3C. Paired t-Tests (voice, SubjectMem, ObjectMem)
    # ────────────────────────────────────────────────────────────
    log("\n── 3C. Paired t-Tests — Voice Effect ──")

    ppv_wide = ppv_ir.pivot(index='participant_ID', columns='Voice', values='HitRate').dropna()
    paired_t_report("Voice -> Hit Rate", ppv_wide, 'A', 'P')

    ppv_wr_df = wr_hits.groupby(['participant_ID', 'Voice'])['Accuracy_WR'].mean().reset_index()
    ppv_wr_wide = ppv_wr_df.pivot(index='participant_ID', columns='Voice', values='Accuracy_WR').dropna()
    paired_t_report("Voice -> WR Accuracy", ppv_wr_wide, 'A', 'P')

    ppv_rt = hits.groupby(['participant_ID', 'Voice'])['RT_IR'].median().reset_index()
    ppv_rt_wide = ppv_rt.pivot(index='participant_ID', columns='Voice', values='RT_IR').dropna()
    paired_t_report("Voice -> RT_IR", ppv_rt_wide, 'A', 'P')

    log("\n── 3C. Paired t-Tests — Subject Memorability (H vs L) ──")

    pps_subj = targets_repeat.groupby(['participant_ID', 'SubjectMem'])['Accuracy_IR'].mean().reset_index()
    pps_subj_wide = pps_subj.pivot(index='participant_ID', columns='SubjectMem', values='Accuracy_IR').dropna()
    paired_t_report("SubjectMem -> Hit Rate", pps_subj_wide, 'H', 'L')

    pps_subj_wr = wr_hits.groupby(['participant_ID', 'SubjectMem'])['Accuracy_WR'].mean().reset_index()
    pps_subj_wr_wide = pps_subj_wr.pivot(index='participant_ID', columns='SubjectMem', values='Accuracy_WR').dropna()
    paired_t_report("SubjectMem -> WR Accuracy", pps_subj_wr_wide, 'H', 'L')

    pps_subj_rt = hits.groupby(['participant_ID', 'SubjectMem'])['RT_IR'].median().reset_index()
    pps_subj_rt_wide = pps_subj_rt.pivot(index='participant_ID', columns='SubjectMem', values='RT_IR').dropna()
    paired_t_report("SubjectMem -> RT_IR", pps_subj_rt_wide, 'H', 'L')

    log("\n── 3C. Paired t-Tests — Object Memorability (H vs L) ──")

    ppo_obj = targets_repeat.groupby(['participant_ID', 'ObjectMem'])['Accuracy_IR'].mean().reset_index()
    ppo_obj_wide = ppo_obj.pivot(index='participant_ID', columns='ObjectMem', values='Accuracy_IR').dropna()
    paired_t_report("ObjectMem -> Hit Rate", ppo_obj_wide, 'H', 'L')

    ppo_obj_wr = wr_hits.groupby(['participant_ID', 'ObjectMem'])['Accuracy_WR'].mean().reset_index()
    ppo_obj_wr_wide = ppo_obj_wr.pivot(index='participant_ID', columns='ObjectMem', values='Accuracy_WR').dropna()
    paired_t_report("ObjectMem -> WR Accuracy", ppo_obj_wr_wide, 'H', 'L')

    ppo_obj_rt = hits.groupby(['participant_ID', 'ObjectMem'])['RT_IR'].median().reset_index()
    ppo_obj_rt_wide = ppo_obj_rt.pivot(index='participant_ID', columns='ObjectMem', values='RT_IR').dropna()
    paired_t_report("ObjectMem -> RT_IR", ppo_obj_rt_wide, 'H', 'L')

    # ────────────────────────────────────────────────────────────
    # 3D. Block Pairwise t-Tests (Hit Rate)
    # ────────────────────────────────────────────────────────────
    log("\n── 3D. Block Pairwise t-Tests (Hit Rate) ──")

    ppb_wide = ppb_ir.pivot(index='participant_ID', columns='Block', values='HitRate').dropna()
    blocks = sorted(ppb_wide.columns)
    block_pairs = [(blocks[i], blocks[j]) for i in range(len(blocks)) for j in range(i + 1, len(blocks))]
    raw_ps = []
    block_results = []
    for b1, b2 in block_pairs:
        if b1 in ppb_wide.columns and b2 in ppb_wide.columns:
            t_stat, p = stats.ttest_rel(ppb_wide[b1], ppb_wide[b2])
            n = len(ppb_wide)
            d = t_stat / np.sqrt(n)
            raw_ps.append(p)
            block_results.append((b1, b2, t_stat, p, d, n))
            log(f"  Block {b1} vs {b2}: t({n-1}) = {t_stat:.4f}, p = {p:.4e}, d = {d:.4f}")

    if len(raw_ps) > 1:
        raw_p_arr = np.array(raw_ps)
        _, bonf_p, _, _ = multipletests(raw_p_arr, method='bonferroni')
        _, holm_p, _, _ = multipletests(raw_p_arr, method='holm')
        log(f"\n  {'Pair':<15} {'raw p':>12} {'Bonferroni':>12} {'Holm':>12}")
        log("  " + "-" * 55)
        for i, (b1, b2, t, p, d, n) in enumerate(block_results):
            sig = ""
            if bonf_p[i] < 0.05: sig += " *Bonf"
            if holm_p[i] < 0.05: sig += " *Holm"
            log(f"  Blk{b1} vs Blk{b2}    {p:>12.4e} {bonf_p[i]:>12.4e} {holm_p[i]:>12.4e}{sig}")

    # ────────────────────────────────────────────────────────────
    # 3E. Post-Hoc Pairwise t-Tests (Conditions) with Corrections
    # ────────────────────────────────────────────────────────────
    log("\n── 3E. Post-Hoc Pairwise t-Tests (Conditions) ──")

    pairs = [(conds[i], conds[j]) for i in range(len(conds)) for j in range(i + 1, len(conds))]

    for dv_label, dv_df, dv_col in [('Hit Rate', ppc_ir, 'HitRate'),
                                     ('WR Accuracy', ppc_wr, 'WR_Rate')]:
        if len(dv_df) == 0:
            continue
        log(f"\n  Pairwise Paired t-Tests: {dv_label}")
        dv_wide = dv_df.pivot(index='participant_ID', columns='Condition', values=dv_col)
        raw_ps = []
        pair_results = []
        for c1, c2 in pairs:
            if c1 in dv_wide.columns and c2 in dv_wide.columns:
                valid = dv_wide[[c1, c2]].dropna()
                if len(valid) >= 2:
                    t_stat, p = stats.ttest_rel(valid[c1], valid[c2])
                    pooled_std = np.sqrt((valid[c1].std() ** 2 + valid[c2].std() ** 2) / 2)
                    d_pooled = (valid[c1].mean() - valid[c2].mean()) / pooled_std if pooled_std > 0 else 0
                    raw_ps.append(p)
                    pair_results.append((c1, c2, t_stat, p, d_pooled, valid[c1].mean(), valid[c2].mean()))

        if len(raw_ps) > 0:
            raw_p_arr = np.array(raw_ps)
            _, bonf_p, _, _ = multipletests(raw_p_arr, method='bonferroni')
            _, holm_p, _, _ = multipletests(raw_p_arr, method='holm')
            _, bh_p, _, _ = multipletests(raw_p_arr, method='fdr_bh')
            log(f"\n  {'Pair':<12} {'t':>8} {'raw p':>12} {'Bonferroni':>12} {'Holm':>12} {'BH-FDR':>12} {'Cohen d':>8}")
            log("  " + "-" * 82)
            for i, (c1, c2, t, p, d_p, m1, m2) in enumerate(pair_results):
                sig = ""
                if bonf_p[i] < 0.05: sig += " *Bonf"
                if holm_p[i] < 0.05: sig += " *Holm"
                if bh_p[i] < 0.05: sig += " *BH"
                log(f"  {c1}-{c2:<7} {t:>8.4f} {p:>12.4e} {bonf_p[i]:>12.4e} {holm_p[i]:>12.4e} {bh_p[i]:>12.4e} {d_p:>8.4f}{sig}")

    # ────────────────────────────────────────────────────────────
    # 3F. Permutation Test for Condition Effect on Hit Rate
    # ────────────────────────────────────────────────────────────
    log("\n── 3F. Permutation Test (Hit Rate by Condition) ──")

    ppc_perm = targets_repeat.groupby(['participant_ID', 'Condition'])['Accuracy_IR'].mean().reset_index()
    ppc_perm_wide = ppc_perm.pivot(index='participant_ID', columns='Condition', values='Accuracy_IR').dropna()

    if len(ppc_perm_wide) >= 5 and all(c in ppc_perm_wide.columns for c in conds):
        observed_means = [ppc_perm_wide[c].mean() for c in conds]
        observed_stat = np.var(observed_means)
        n_perm = 5000
        all_vals = ppc_perm_wide[conds].values
        perm_stats = []
        for _ in range(n_perm):
            shuffled = np.array([np.random.permutation(row) for row in all_vals])
            perm_stats.append(np.var(shuffled.mean(axis=0)))
        perm_stats = np.array(perm_stats)
        perm_p = (np.sum(perm_stats >= observed_stat) + 1) / (n_perm + 1)
        log(f"  Observed variance of condition means: {observed_stat:.6f}")
        log(f"  Permutation p-value ({n_perm} permutations): {perm_p:.4f}")
        log(f"  {'*** SIGNIFICANT ***' if perm_p < 0.05 else 'Not significant'}")

    # ────────────────────────────────────────────────────────────
    # 3G. Block × Condition Interaction (KW per block)
    # ────────────────────────────────────────────────────────────
    log("\n── 3G. Block × Condition: Kruskal-Wallis per Block ──")

    for dv_label, dv_src, dv_col in [("Hit Rate", targets_repeat, 'Accuracy_IR'),
                                      ("WR Accuracy", wr_hits, 'Accuracy_WR')]:
        if len(dv_src) == 0:
            continue
        log(f"\n  {dv_label}:")
        for b in sorted(dv_src['Block'].unique()):
            blk_data = dv_src[dv_src['Block'] == b]
            ppc = blk_data.groupby(['participant_ID', 'Condition'])[dv_col].mean().reset_index()
            groups = [ppc[ppc['Condition'] == c][dv_col].values for c in conds]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                h, p = stats.kruskal(*groups)
                log(f"    Block {b}: H = {h:.4f}, p = {p:.4e} {'***' if p < 0.05 else ''}")

    # ────────────────────────────────────────────────────────────
    # 3H. Exclusion Summary
    # ────────────────────────────────────────────────────────────
    log("\n── 3H. Participant / Block Exclusion Summary ──")
    total_blocks = len(val_results)
    passed = val_results[val_results['Passed'] == True]
    failed = val_results[val_results['Passed'] == False]
    log(f"  Total blocks: {total_blocks}")
    log(f"  Passed validation: {len(passed)} ({100*len(passed)/total_blocks:.1f}%)")
    log(f"  Failed validation: {len(failed)} ({100*len(failed)/total_blocks:.1f}%)")
    n_retained = df_valid['participant_ID'].nunique()
    log(f"  Participants retained: {n_retained}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    log("=" * 70)
    log("  SENTENCE MEMORABILITY — ANALYSIS (REPORT VERSION)")
    log("  Phase 1: Data Loading & Preparation     (Member 1)")
    log("  Phase 2: Descriptive Stats & Plots       (Member 2)")
    log("  Phase 3: Inferential Statistics           (Member 3)")
    log("=" * 70)

    (df_valid, val_results, targets_repeat, hits,
     false_alarms, per_sentence_ir, per_sentence_wr, fa_rate_df) = phase1_load()

    (ppc_ir, ppc_rt, ppc_wr, ppv_ir, ppb_ir, ppv_wr, ppb_wr,
     pps, ppc_corrected_ir, ppt_corr) = phase2_descriptive(
        targets_repeat, hits, per_sentence_ir, per_sentence_wr, fa_rate_df)

    phase3_inferential(targets_repeat, hits, ppc_ir, ppc_rt, ppc_wr,
                       ppv_ir, ppb_ir, fa_rate_df, val_results, df_valid)

    save_results()
    log("\nDone — 11 plots saved to figures/, results saved to analysis_results.txt")
