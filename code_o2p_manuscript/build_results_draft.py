# -*- coding: utf-8 -*-
"""⑫ o2pulse_flattening — Results 초안 · Table 1 · Table 2를 PI 재실행 결과 CSV에서 만든다(손으로 수치를 옮기지 않는다).
실행: python build_results_draft.py <results_o2p_v3_pi> <results_o2p_ai_v2_pi> <out_dir>
수치 출처(사용자 결정 2026-10-22): PI 재실행본. 보고 원칙: 사전지정 결과는 통과·미달 모두 본문에, 사후는 '사후'로 표시.
"""
import sys, os, json
import numpy as np, pandas as pd

V3, AI, OUT, V12 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]   # V12 = results_o2p_v2/v1_vs_v2.csv (사후 제외 전후)
os.makedirs(OUT, exist_ok=True)
r = lambda f: pd.read_csv(os.path.join(V3, f))
ra = lambda f: pd.read_csv(os.path.join(AI, f))

def pct(x, d=1): return f'{100 * x:.{d}f}'
def ci_pct(lo, hi, d=1): return f'{pct(lo, d)} to {pct(hi, d)}'
def f2(x): return f'{x:.2f}'.replace('-', '−')
def ci2(lo, hi): return f'{f2(lo)} to {f2(hi)}'
def pv(p):
    if p < 0.001: return '< 0.001'
    return f'= {p:.3f}' if p < 0.1 else f'= {p:.2f}'

# ── 표본 ──────────────────────────────────────────────
FL = r('sample_flow.csv').set_index('step')
g = lambda s, c='tests': int(FL.loc[s, c])
T = r('per_test.csv')
E = T[(T.fail.fillna('') == '') & (T.age >= 18) & (T.main_evaluable == True) & (T.sealed == True)]
P1S = E.sort_values(['subj', 'age', 'id_test']).drop_duplicates('subj')
PR = E.sort_values(['subj', 'age', 'id_test']).groupby('subj').filter(lambda d: len(d) >= 2).groupby('subj').head(2)
assert len(P1S) == g('P1_sealed_first_tests') and PR.subj.nunique() == g('P2_sealed_pairs')
gap = PR.groupby('subj').age.agg(lambda a: a.iloc[1] - a.iloc[0])
excl_total = g('excl_speed_channel_anomaly') + g('excl_user_signal_741_1_539_1')
VV = pd.read_csv(V12); vv = lambda f, st: VV[(VV.file == f) & (VV.stat == st)].iloc[0]
OLS = r('outliers.csv'); OLP = OLS[OLS.id_test.isin(P1S.id_test)]
INF0 = r('S2_influence.csv')
assert vv('S1_vo2plateau.csv', 'F_neg_n').v1 - vv('S1_vo2plateau.csv', 'F_neg_n').v2 == 2 and vv('S1_vo2plateau.csv', 'F_pos_n').v1 == vv('S1_vo2plateau.csv', 'F_pos_n').v2
assert vv('P2_kappa.csv', 'n_pairs').v1 == vv('P2_kappa.csv', 'n_pairs').v2   # 제외 2건 = 첫 검사 · 평탄화 음성 · 쌍 아님

# ── 일차 · 이차 ───────────────────────────────────────
P1 = r('P1_prevalence.csv').iloc[0]; P2 = r('P2_kappa.csv').iloc[0]
S1 = r('S1_vo2plateau.csv').set_index('group'); S2 = r('S2_logistic.csv').set_index('term')
S3 = r('S3_definitions.csv').set_index('definition'); K3 = r('S3_def_kappa.csv')
SE = r('sensitivity.csv'); PH = r('posthoc.csv'); RS = r('ramp_strata.csv'); G4 = r('G4_strata.csv')
SR = r('S2_ramp_adjusted.csv').set_index('term'); INF = r('S2_influence.csv')
se = lambda i, a: SE[(SE.id == i) & (SE.analysis == a)].iloc[0]
ph = lambda i, a: PH[(PH.id == i) & (PH.analysis == a)].iloc[0]
k3 = K3[(K3.def1 != K3.def2)]
k3u = k3[k3.def1 < k3.def2]
k3p = k3[k3.def1 == 'D120 (primary)']
pos, neg = S1.loc['F_pos'], S1.loc['F_neg']
st = lambda s, l: G4[(G4.stratifier == s) & (G4.level == l)].iloc[0]
rs = RS.set_index('ramp')

# ── AI ────────────────────────────────────────────────
A1 = ra('A1_summary.csv').set_index('model'); C1 = ra('A1_compare.csv').set_index('model')
A3 = ra('A3_summary.csv').set_index('model'); C3 = ra('A3_compare.csv').set_index('model')
A2 = ra('A2_summary.csv'); A4 = ra('A4_repeat.csv').iloc[0]; IMP = ra('importance.csv')
PHA = ra('posthoc_ai.csv'); AFL = ra('sample_flow.csv').set_index('step')
a2 = lambda t, m: A2[(A2.tier == t) & (A2.model == m)].iloc[0]
imp1 = IMP[IMP.task == 'A1'].sort_values('importance', ascending=False)
bic = PHA[PHA.analysis == 'A4 BIC k'].copy(); bic['k'] = bic.k.astype(int); kmin = int(bic.loc[bic.value.idxmin(), 'k'])
FEAT = {'speed_slope': 'the rate of speed increase over the preceding 2 min', 'elapsed_c': 'elapsed exercise time',
        'rer_c': 'the respiratory exchange ratio', 'o2p_slope': 'the preceding O₂ pulse slope'}
top = [FEAT.get(f, f) for f in imp1.feature.head(3)]

SGr = ra('subgroup.csv'); SGa = SGr[SGr.task == 'A1']; SGm = SGr[SGr.task == 'A3']
ap = lambda m: f"{f2(A1.loc[m,'auc'])} ({ci2(A1.loc[m,'lo'], A1.loc[m,'hi'])})"
L = []; A = L.append
A('# Results — draft')
A('')
A('<!-- 자동 생성: code_o2p_manuscript/build_results_draft.py · 수치 출처 = PI 재실행본 results_o2p_v3_pi · results_o2p_ai_v2_pi (사용자 결정 2026-10-22) · 그림 번호 잠정. 손으로 고치지 말고 코드를 고친다. -->')
A('')
A('## Participants and analytic samples')
A('')
A(f"Of {g('tests_total')} tests from {g('tests_total','participants')} participants, {excl_total} were excluded for signal artefacts, "
  f"{g('excl_age_lt18')} were from participants younger than 18 years and {g('excl_not_evaluable_main')} had fewer than eight valid 10-s bins in the final 120 s, "
  f"leaving {g('evaluable_adult')} evaluable tests from {g('evaluable_adult','participants')} adults. The screening subset set aside "
  f"{g('screen_30pct_set_aside')} tests from {g('screen_30pct_set_aside','participants')} participants, and the sealed subset contained "
  f"{g('sealed_tests')} tests from {g('sealed_tests','participants')} participants. The prevalence analysis used the first test of each of these "
  f"{len(P1S)} participants ({int((P1S.sex == 0).sum())} men, {int((P1S.sex == 1).sum())} women; Table 1), and {PR.subj.nunique()} participants contributed a test–retest pair "
  f"(median interval {gap.median():.1f} years, IQR {gap.quantile(.25):.1f}–{gap.quantile(.75):.1f}, from the recorded ages).")
A('')
A(f"The two tests excluded for implausible gas signals were identified after the first analysis of the sealed subset (post hoc exclusion); both were first tests without flattening and neither belonged to a test–retest pair. "
  f"Before their exclusion, prevalence was {pct(vv('P1_prevalence.csv','p').v1)}% (95% CI {ci_pct(vv('P1_prevalence.csv','lo').v1, vv('P1_prevalence.csv','hi').v1)}; n = {int(vv('P1_prevalence.csv','n').v1)}) "
  f"and the odds ratio for the peak respiratory exchange ratio {f2(vv('S2_logistic.csv','rer01_OR').v1)} ({ci2(vv('S2_logistic.csv','rer01_lo').v1, vv('S2_logistic.csv','rer01_hi').v1)}); κ was unchanged.")
A('')
A('## Prevalence and test–retest agreement (Fig. 1)')
A('')
A(f"The O₂ pulse slope over the final 120 s was ≤ 0 in {int(P1.k)} of {int(P1.n)} first tests, a prevalence of {pct(P1.p)}% (95% CI {ci_pct(P1.lo, P1.hi)}; Fig. 1a, b). "
  f"The lower confidence limit exceeded the pre-specified 10% threshold, and the interval lay above both the 20% reported in elite soccer players {{Perim, 2011}} "
  f"and the 35% reported in healthy untrained adults, in whom flattening was counted at any point of the test {{Mainzer, 2026}}.")
A('')
A(f"In the {int(P2.n_pairs)} test–retest pairs, both tests were classified as flattening in {int(P2.both_pos)} pairs, neither in {int(P2.both_neg)}, and the classification changed in "
  f"{int(P2.discordant)} (raw agreement {pct(P2.agree, 0)}%; Fig. 1c). Cohen κ was {f2(P2.kappa)} (95% CI {ci2(P2.lo, P2.hi)}); because the upper limit was below 0.40, "
  f"flattening could not be regarded as a reproducible characteristic of the individual in these data.")
A('')
A('## Robustness and sensitivity analyses (Fig. 2)')
A('')
w60, w180, b30 = se('SE2', 'P1'), se('SE3', 'P1'), se('SE1', 'P1')
p1r = SE[SE.analysis == 'P1']
LBL = {'SE3': '180-s window, below 35%', 'SE8': 'standard ramp only, including 35%'}
not_above = [LBL.get(x.id, x.label) + ('' if x.id in LBL else (', below 35%' if x.hi < .35 else ', including 35%')) for x in p1r.itertuples() if x.lo <= .35]
A(f"Prevalence depended on the length of the analysis window: {pct(w60.est)}% (95% CI {ci_pct(w60.lo, w60.hi)}) with 60 s and {pct(w180.est)}% "
  f"({ci_pct(w180.lo, w180.hi)}) with 180 s, compared with {pct(b30.est)}% ({ci_pct(b30.lo, b30.hi)}) when 30-s instead of 10-s bins were used over 120 s (Fig. 2). "
  f"Including participants younger than 18 years ({pct(se('SE4','P1').est)}%, n = {int(se('SE4','P1').n)}), all sealed tests with participant-clustered intervals "
  f"({pct(se('SE5','P1').est)}%, n = {int(se('SE5','P1').n)} tests), the full data set including the screening subset ({pct(se('SE6','P1').est)}%, n = {int(se('SE6','P1').n)}) "
  f"and only tests with a peak respiratory exchange ratio ≥ 1.05 ({pct(se('SE7','P1').est)}%, n = {int(se('SE7','P1').n)}) gave similar estimates. "
  f"The lower confidence limit exceeded 10% in every analysis{'' if (p1r.lo > .10).all() else ' [CHECK]'}; the interval lay entirely above 35% in all but "
  f"{len(not_above)} analyses ({'; '.join(not_above)}).")
A('')
p2r = SE[(SE.analysis == 'P2') & SE.est.notna()]
A(f"Across the pre-specified sensitivity analyses, κ ranged from {f2(p2r.est.min())} to {f2(p2r.est.max())} and no upper confidence limit reached 0.40 "
  f"(highest {f2(p2r.hi.max())}; n = {int(p2r.n.min())}–{int(p2r.n.max())} pairs). Two post hoc analyses led to the same conclusions: excluding tests with a peak respiratory exchange ratio "
  f"outside 0.9–1.6 ({pct(ph('PH1','P1').est)}%, n = {int(ph('PH1','P1').n)}; κ {f2(ph('PH1','P2').est)}) and excluding all flagged outlier candidates "
  f"({pct(ph('PH2','P1').est)}%, n = {int(ph('PH2','P1').n)}; κ {f2(ph('PH2','P2').est)}). The outlier rules flagged {OLP.id_test.nunique()} of the {len(P1S)} first tests "
  f"({int((OLP.rule.str.startswith('robust')).sum())} robust z-score flags and {int((OLP.rule.str.startswith('valid')).sum())} values outside the plausible ranges, "
  f"{int((OLP.rule.str.startswith('valid') & (OLP.variable == 'rer_peak')).sum())} of them for the peak respiratory exchange ratio).")
A('')
A('## Sex, age and the rate of speed increase (Fig. 3)')
A('')
A(f"Prevalence was {pct(st('sex','Male').p)}% in men (n = {int(st('sex','Male').n)}) and {pct(st('sex','Female').p)}% in women (n = {int(st('sex','Female').n)}), "
  f"and {pct(st('age','18–29').p)}%, {pct(st('age','30–39').p)}% and {pct(st('age','≥40').p)}% in participants aged 18–29, 30–39 and ≥ 40 years (Fig. 3a, b).")
A('')
sd, sl, fa = rs.loc['standard (0.98–1.02]'], rs.loc['slow (≤0.98)'], rs.loc['fast (>1.02)']
s8 = se('SE8', 'P1'); s8k = se('SE8', 'P2')
A(f"In the data-informed protocol analyses, speed increased in the final 120 s at > 0.98 to ≤ 1.02 km·h⁻¹·min⁻¹ in {int(sd.n)} of the {int(P1.n)} first tests, more slowly in {int(sl.n)} and faster in {int(fa.n)}. "
  f"Prevalence was {pct(sd.p)}% (95% CI {ci_pct(sd.lo, sd.hi)}) with the standard ramp, {pct(sl.p)}% ({ci_pct(sl.lo, sl.hi)}) with a slower increase and "
  f"{pct(fa.p)}% ({ci_pct(fa.lo, fa.hi)}) with a faster increase (Fig. 3c). Restricted to the standard ramp, the lower limit still exceeded 10% and 20%, but the interval included 35%; the slower category also contained tests in which speed was held almost constant over the final 120 s (lowest rate {abs(P1S.ramp_last120.min()) if abs(P1S.ramp_last120.min()) < .005 else P1S.ramp_last120.min():.2f} km·h⁻¹·min⁻¹); "
  f"κ in the {int(s8k.n)} pairs in which both tests used the standard ramp was {f2(s8k.est)} (95% CI {ci2(s8k.lo, s8k.hi)}).")
A('')
A('## Factors associated with flattening (Table 2)')
A('')
rer = S2.loc['rer01']; age = S2.loc['age10']
oth = [t for t in ['age10', 'sex', 'vo2kg5', 'phr5', 'spd1']]
A(f"In the pre-specified model, a higher peak respiratory exchange ratio was associated with flattening (odds ratio {f2(rer.OR)} per 0.1, 95% CI {ci2(rer.lo, rer.hi)}; "
  f"Holm-adjusted P {pv(rer.p_holm)}). For age the odds ratio was {f2(age.OR)} per 10 years ({ci2(age.lo, age.hi)}; unadjusted P {pv(age.p)}, Holm-adjusted P {pv(age.p_holm)}); "
  f"the intervals for sex, VO₂peak, percentage of predicted maximal heart rate and peak speed included 1 (Holm-adjusted P {pv(min(S2.loc[t,'p_holm'] for t in ['sex','vo2kg5','phr5','spd1']))}). "
  f"Without the {int(INF.flag_cook.sum())} observations with Cook's distance > 4/n, the odds ratio for the respiratory exchange ratio was {f2(rer.OR_wo_cook)} "
  f"({ci2(rer.lo_wo_cook, rer.hi_wo_cook)}) and that for age {f2(age.OR_wo_cook)} ({ci2(age.lo_wo_cook, age.hi_wo_cook)}; Holm-adjusted P {pv(age.p_holm_wo_cook)}). "
  f"Leverage flagged {int(INF0.flag_lev.sum())} observations, DFBETAS {int(INF0.flag_dfbeta.sum())} and the Bonferroni residual limit {int(INF0.flag_resid.sum())}.")
A('')
rr = SR.loc['rer01']
A(f"With the ramp category added (data-informed sensitivity analysis), the odds ratio for the respiratory exchange ratio was {f2(rr.OR)} ({ci2(rr.lo, rr.hi)}; Holm-adjusted P {pv(rr.p_holm)}); "
  f"relative to the standard ramp, the odds of flattening were higher with a slower ({f2(SR.loc['ramp_slow','OR'])}, {ci2(SR.loc['ramp_slow','lo'], SR.loc['ramp_slow','hi'])}) "
  f"and a faster increase ({f2(SR.loc['ramp_fast','OR'])}, {ci2(SR.loc['ramp_fast','lo'], SR.loc['ramp_fast','hi'])}). In the post hoc analysis excluding tests with a peak respiratory "
  f"exchange ratio outside 0.9–1.6, the odds ratio from the pre-specified model was {f2(ph('PH1','S2 rer01 OR').est)} ({ci2(ph('PH1','S2 rer01 OR').lo, ph('PH1','S2 rer01 OR').hi)}).")
A('')
A('## Definition dependence and accompanying responses (Figs. 4 and 5)')
A('')
dd = S3.drop('D120 (primary)')
A(f"The four alternative definitions classified {pct(dd.p.min())}% to {pct(dd.p.max())}% of the same {int(P1.n)} tests as flattening, compared with {pct(S3.loc['D120 (primary)','p'])}% "
  f"for the primary definition (Fig. 4a). Agreement between the primary and alternative definitions was κ {f2(k3p.kappa.min())}–{f2(k3p.kappa.max())} (Fig. 4b), "
  f"and between any two of the five definitions {f2(k3u.kappa.min())}–{f2(k3u.kappa.max())}.")
A('')
A(f"A VO₂ slope ≤ 0 over the same 120 s accompanied {int(pos.vo2_plateau)} of {int(pos.n)} tests with flattening ({pct(pos.p)}%, 95% CI {ci_pct(pos.lo, pos.hi)}) and "
  f"{int(neg.vo2_plateau)} of {int(neg.n)} tests without ({pct(neg.p)}%, {ci_pct(neg.lo, neg.hi)}). The median heart rate slope over the final 120 s was "
  f"{pos.hr_slope_median:.1f} beats·min⁻¹ per min (IQR {pos.hr_slope_q1:.1f}–{pos.hr_slope_q3:.1f}) in tests with flattening and {neg.hr_slope_median:.1f} "
  f"({neg.hr_slope_q1:.1f}–{neg.hr_slope_q3:.1f}) in tests without (Fig. 5).")
A('')
A('## Secondary machine-learning analyses (Fig. 6)')
A('')
A(f"These analyses used all {int(AFL.loc['A1 / A2 tests','n'])} evaluable adult tests from {int(AFL.loc['A1 / A2 participants','n'])} participants with participant-grouped cross-validation. "
  f"Information available 2 min before the end of exercise predicted flattening better than chance but weakly: area under the curve {ap('LR')} for logistic regression, "
  f"{ap('RF')} for random forest, {ap('HGB')} for gradient boosting and {ap('MLP')} for the multilayer perceptron (Fig. 6a). No machine-learning model was better than logistic regression "
  f"after Holm adjustment (random forest − logistic {C1.loc['RF','diff']:.3f}, 95% CI {C1.loc['RF','lo']:.4f} to {C1.loc['RF','hi']:.3f}; unadjusted P {pv(C1.loc['RF','p'])}, Holm-adjusted P {pv(C1.loc['RF','p_holm'])}). "
  f"The most informative inputs of the random forest (permutation importance) were {top[0]}, {top[1]} and {top[2]}. Performance was similar in men and women and across age groups "
  f"(area under the curve {f2(SGa.est.min())}–{f2(SGa.est.max())}).")
A('')
A(f"Peak O₂ pulse was estimated from heart rate, speed and anthropometric data with a mean absolute error of {f2(A3.loc['Linear','mae'])} mL·beat⁻¹ "
  f"(95% CI {ci2(A3.loc['Linear','mae_lo'], A3.loc['Linear','mae_hi'])}; R² {f2(A3.loc['Linear','r2'])}) by linear regression, against {f2(A3.loc['Dummy','mae'])} for the mean-only model; "
  f"no machine-learning model was more accurate (Holm-adjusted P ≥ {f2(C3.p_holm.min())}; Fig. 6b); for the random forest the error was {f2(SGm[SGm.level=='Female'].est.iloc[0])} mL·beat⁻¹ in women and {f2(SGm[SGm.level=='Male'].est.iloc[0])} in men. Clustering the shape of the whole-test O₂ pulse curve selected the largest "
  f"pre-specified number of clusters (k = {int(A4.k_best)}), and a post hoc extension showed that the information criterion kept decreasing up to k = {kmin}; "
  f"cluster membership agreed poorly between repeat tests (κ {f2(A4.kappa)}, 95% CI {ci2(A4.lo, A4.hi)}; {int(A4.n_pairs)} pairs).")
A('')
hs, hr_ = a2('HR+speed', 'GRU'), a2('HR+speed+resp', 'GRU')
kmax_hs = A2[A2.tier == 'HR+speed'].hi.max()
A(f"Flattening classified from O₂ pulse curves reconstructed without gas analysis agreed poorly with flattening classified from measured curves. With heart rate and speed as inputs, "
  f"κ was {f2(A2[A2.tier=='HR+speed'].kappa.min())}–{f2(A2[A2.tier=='HR+speed'].kappa.max())} across ridge regression, the convolutional network and the recurrent network "
  f"(upper limits ≤ {f2(kmax_hs)}). Adding respiratory rate and minute ventilation raised κ to {f2(hr_.kappa)} (95% CI {ci2(hr_.lo, hr_.hi)}) for the recurrent network, "
  f"an interval that included 0.40, so agreement could not be judged; the other two models remained below 0.40 (Fig. 6c). "
  f"The mean absolute error of the reconstructed oxygen uptake was {hr_.vo2_mae:.0f} mL·min⁻¹ over the whole test and {hr_.vo2_mae_last120:.0f} mL·min⁻¹ over the final 120 s for this model, "
  f"and {A2.vo2_mae.min():.0f}–{A2.vo2_mae.max():.0f} mL·min⁻¹ over the whole test across the six models.")
A('')


# ── Table 1 · Table 2 ─────────────────────────────────
def msd(x, d=1): return f'{x.mean():.{d}f} ± {x.std():.{d}f}'
grp = {'All': P1S, 'Flattening': P1S[P1S.main_F == True], 'No flattening': P1S[P1S.main_F == False]}
rows = [('n', lambda d: f'{len(d)}'), ('Women, n (%)', lambda d: f"{int((d.sex == 1).sum())} ({100 * (d.sex == 1).mean():.0f})"),
        ('Age, years', lambda d: msd(d.age)), ('Body mass, kg', lambda d: msd(d.weight)),
        ('VO₂peak, mL·kg⁻¹·min⁻¹', lambda d: msd(d.vo2peak_kg)), ('Maximal heart rate, beats·min⁻¹', lambda d: msd(d.hrmax, 0)),
        ('Maximal heart rate, % predicted', lambda d: msd(d.phrmax)), ('Peak respiratory exchange ratio', lambda d: msd(d.rer_peak, 2)),
        ('Peak speed, km·h⁻¹', lambda d: msd(d.speed_max)), ('Peak O₂ pulse, mL·beat⁻¹', lambda d: msd(d.o2p_max)),
        ('Exercise duration, min', lambda d: msd(d.t_end / 60)),
        ('Speed increase in final 120 s, km·h⁻¹·min⁻¹', lambda d: f"{d.ramp_last120.median():.2f} [{d.ramp_last120.quantile(.25):.2f}–{d.ramp_last120.quantile(.75):.2f}]")]
T1 = pd.DataFrame([{'Characteristic': n, **{k: fn(v) for k, v in grp.items()}} for n, fn in rows])
T1.to_csv(os.path.join(OUT, 'Table1_characteristics.csv'), index=False)
LAB = {'age10': 'Age (per 10 years)', 'sex': 'Female (vs male)', 'vo2kg5': 'VO₂peak (per 5 mL·kg⁻¹·min⁻¹)', 'phr5': 'Maximal heart rate (per 5% of predicted)',
       'rer01': 'Peak respiratory exchange ratio (per 0.1)', 'spd1': 'Peak speed (per 1 km·h⁻¹)', 'ramp_slow': 'Slower ramp (vs standard)', 'ramp_fast': 'Faster ramp (vs standard)'}
fp = lambda p: '< 0.001' if p < .001 else f'{p:.3f}'
T2 = []
for t in LAB:
    a = S2.loc[t] if t in S2.index else None; b = SR.loc[t]
    T2.append({'Variable': LAB[t], 'Model 1 OR (95% CI)': f"{f2(a.OR)} ({ci2(a.lo, a.hi)})" if a is not None else '—',
               'Model 1 P (Holm)': fp(a.p_holm) if a is not None else '—',
               'Model 2 OR (95% CI)': f"{f2(b.OR)} ({ci2(b.lo, b.hi)})", 'Model 2 P (Holm)': fp(b.p_holm) if pd.notna(b.p_holm) else f"{fp(b.p)}†"})
T2 = pd.DataFrame(T2); T2.to_csv(os.path.join(OUT, 'Table2_logistic.csv'), index=False)

def md(df): return '\n'.join(['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)] + ['| ' + ' | '.join(map(str, r_)) + ' |' for r_ in df.values])
A('## Table 1')
A('')
A(f"Table 1. Characteristics of the participants in the prevalence analysis (first sealed test, n = {len(P1S)}).")
A('')
A(md(T1))
A('')
A('Exercise duration is the time from the start of the recording, including the warm-up, to the end of exercise. Values are mean ± SD, n (%) or median [IQR]. Abbreviations: IQR, interquartile range; SD, standard deviation; VO₂peak, peak oxygen uptake.')
A('')
A('## Table 2')
A('')
A(f"Table 2. Factors associated with O₂ pulse flattening (n = {int(rer.n)} tests, {int(rer.events)} with flattening).")
A('')
A(md(T2))
A('')
A('Model 1 is the pre-specified model; Model 2 adds the ramp category (sensitivity analysis added after an exploratory observation). † Unadjusted P; ramp terms were not part of the Holm family. '
  'Values are odds ratios (95% CI) from logistic regression with Wald intervals; P values are Holm-adjusted across the six pre-specified factors. '
  'Abbreviations: CI, confidence interval; OR, odds ratio; VO₂peak, peak oxygen uptake.')
A('')
A('## Figure legends (draft)')
A('')
A(f"Figure 1. O₂ pulse flattening in the final 120 s and its test–retest agreement. (a) O₂ pulse slope of every analysed test by sex (points with violin), with flattening (slope ≤ 0) marked and the mean slope (95% CI) shown as a diamond. "
  f"(b) Prevalence with reference lines at 10% (decision threshold), 20% {{Perim, 2011}} and 35% {{Mainzer, 2026}}. (c) Slopes of the first two tests in the {int(P2.n_pairs)} test–retest pairs; "
  f"coloured lines mark pairs whose classification changed. Values are % (Wilson 95% CI) in (b) and Cohen κ (95% CI, participant bootstrap, 2,000 resamples) in (c). "
  f"Abbreviations: CI, confidence interval.")
A('')
A("Figure 2. Robustness of prevalence and test–retest agreement. Filled circles, sensitivity analyses (pre-specified except the standard-ramp analysis, added after an exploratory observation); dashed lines, decision thresholds (10% and κ 0.40); open circles, post hoc analyses; n is tests (prevalence) or pairs (κ). "
  "Values are % (Wilson 95% CI; participant-clustered bootstrap CI for all sealed tests) and Cohen κ (95% CI, participant bootstrap). Abbreviations: CI, confidence interval; RER, respiratory exchange ratio.")
A('')
A("Figure 3. Prevalence of flattening by (a) sex, (b) age group and (c) rate of speed increase in the final 120 s (data-informed analysis). Jittered points at 0 and 100% are individual tests. Values are % (Wilson 95% CI) with n per group. "
  "Abbreviations: CI, confidence interval.")
A('')
A("Figure 4. Dependence on the definition of flattening. (a) Prevalence under five definitions in the same tests: D120, primary (slope ≤ 0 over the final 120 s); Dthird, slope ≤ 0 over the last third; Dratio, last-third slope ≤ 25% of the middle-third slope; Ddecl, early peak with a final decline; Dci, upper 95% limit of the primary slope ≤ 0. (b) Agreement of each alternative definition with the primary definition. Values are % (Wilson 95% CI) in (a) "
  "and Cohen κ (bootstrap 95% CI, one test per participant) in (b). Abbreviations: CI, confidence interval.")
A('')
A("Figure 5. (a) O₂ pulse, (b) VO₂ and (c) heart rate over the final 120 s in tests with and without flattening. Grey lines are individual tests; coloured lines are group means of 10-s bins. Values are mean (95% CI). Abbreviations: HR, heart rate; VO₂, oxygen uptake.")
A('')
A("Figure 6. Secondary machine-learning analyses. (a) Prediction of flattening 2 min before the end of exercise. (b) Estimation of peak O₂ pulse without gas analysis. "
  "(c) Agreement of flattening classified from reconstructed and measured O₂ pulse curves. Points are outer folds (5 × 5 participant-grouped cross-validation) in (a) and (b). "
  "Values are mean (95% CI, Nadeau–Bengio corrected) in (a) and (b) and Cohen κ (95% CI, participant bootstrap) in (c). Abbreviations: AUC, area under the receiver operating "
  "characteristic curve; CI, confidence interval; CNN, convolutional neural network; GRU, gated recurrent unit; HGB, histogram gradient boosting; LR, logistic regression; "
  "MAE, mean absolute error; MLP, multilayer perceptron; RF, random forest.")
open(os.path.join(OUT, 'results_draft.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
body = '\n'.join(l for l in L if not l.startswith('|') and not l.startswith('<!--'))
i0, i1 = body.index('## Participants'), body.index('## Table 1')
import re
words = len(re.sub(r'\{[^}]*\}', '', body[i0:i1]).split())
print('Results words (excl. tables/legends, temp citations):', words)
