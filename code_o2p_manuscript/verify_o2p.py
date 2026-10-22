# -*- coding: utf-8 -*-
"""⑫ 원고 수치 ↔ 결과 CSV 기계 대조 (자체 게이트 1).
실행: python verify_o2p.py <results_o2p_v3_pi> <results_o2p_ai_v2_pi> <manuscript_dir>
본문(Results · Methods)의 모든 수 토큰이 (a) 결과 CSV 셀을 어떤 반올림 · 백분율로 나타낸 값이거나 (b) per_test.csv에서 독립 재계산한 기술통계이거나
(c) Methods 고정 상수 목록에 있어야 한다. 하나라도 없으면 exit 1.
"""
import sys, os, re, glob
import numpy as np, pandas as pd

V3, AI, MS = sys.argv[1:4]
V12 = sys.argv[4] if len(sys.argv) > 4 else None   # results_o2p_v2/v1_vs_v2.csv (사후 제외 전 값)
ok = set()
def add(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return
    x = float(x)
    for v in (x, 100 * x, abs(x), 100 * abs(x), x / 60):
        for d in range(0, 5): ok.add(f'{round(v, d):.{d}f}')
for f in glob.glob(os.path.join(V3, '*.csv')) + glob.glob(os.path.join(AI, '*.csv')) + ([V12] if V12 else []):
    if os.path.basename(f) in ('per_test.csv', 'S2_influence.csv'): continue
    d = pd.read_csv(f)
    for c in d.columns:
        for v in pd.to_numeric(d[c], errors='coerce').dropna(): add(v)
# per_test 독립 재계산(빌더와 다른 구현: 인덱스 · groupby 사용)
T = pd.read_csv(os.path.join(V3, 'per_test.csv'))
m = (T.fail.isna() | (T.fail == '')) & (T.age >= 18) & (T.main_evaluable.astype(str) == 'True') & (T.sealed.astype(str) == 'True')
E = T[m].sort_values(['subj', 'age', 'id_test'])
first = E.groupby('subj', sort=False).nth(0)
two = E.groupby('subj').filter(lambda g: len(g) > 1).groupby('subj').nth[:2]
gap = two.groupby('subj').age.apply(lambda a: a.max() - a.min())
for x in [len(first), gap.median(), gap.quantile(.25), gap.quantile(.75)]: add(x)
for grp in [first, first[first.main_F.astype(str).isin(['True', '1.0'])], first[first.main_F.astype(str).isin(['False', '0.0'])]]:
    add(len(grp)); add((grp.sex == 1).sum()); add((grp.sex == 0).sum()); add(100 * (grp.sex == 1).mean() / 100)
    for c in ['age', 'weight', 'vo2peak_kg', 'hrmax', 'phrmax', 'rer_peak', 'speed_max', 'o2p_max']:
        add(grp[c].mean()); add(grp[c].std())
    add((grp.t_end / 60).mean()); add((grp.t_end / 60).std())
    for q in (.25, .5, .75): add(grp.ramp_last120.quantile(q))
INF = pd.read_csv(os.path.join(V3, 'S2_influence.csv')); add(INF.flag_cook.sum())
# Methods 고정 상수(사전지정 매개변수 · 표기)
CONST = set('''0 1 2 3 4 5 6 8 10 12 15 18 20 25 30 32 35 45 60 64 90 95 100 120 180 208 230 857 992 2008 2018 2000 2,000
0.5 0.7 0.9 1.6 0.98 1.02 1.05 0.40 0.1 0.6745 3.5 0.05 1.0.1 3.12.10 0.096 2.4.6 3.0.3 1.18.0 0.15.0 1.9.0 2.21.0 3.15.1 256 3 4 7 9 11'''.split())
NUM = re.compile(r'(?<![A-Za-z0-9_.])[−-]?\d[\d,]*(?:\.\d+)*')
bad = []
for fn in ['results_draft.md', 'methods_draft.md']:
    txt = open(os.path.join(MS, fn), encoding='utf-8').read()
    txt = re.sub(r'<!--.*?-->', '', txt, flags=re.S)
    txt = txt.split('## References')[0]                    # 서지 목록은 대조 대상 아님(인용 게이트가 본다)
    txt = re.sub(r'\{[^}]*\}', '', txt)                  # 임시 인용
    txt = re.sub(r'(Fig(?:ure)?s?\.?|Table|SE|PH|P|S|A|D|Model|k =|SHA-)\s?\d+', '', txt)  # 번호 · 식별자
    txt = re.sub(r'[A-Za-z]+\d+', '', txt); txt = re.sub(r'\d+-s|\d+-fold|\d+-breath|\d+-lead|\d+-D', '', txt)
    for line_no, line in enumerate(txt.splitlines(), 1):
        if line.startswith('#'): continue
        for tok in NUM.findall(line):
            t = tok.replace('−', '-').lstrip('-').rstrip('.')
            if t in CONST or t.replace(',', '') in CONST or t in ok or t.replace(',', '') in ok: continue
            bad.append((fn, line_no, tok, line[:90]))
print(f'number tokens checked; unmatched = {len(bad)}')
for b in bad: print(' ', b)
sys.exit(1 if bad else 0)
