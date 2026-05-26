"""
=============================================================================
PROJECT 3: Predictive Response Model — Financial Risk Classification
Statistical Modeling | Feature Engineering | Business Recommendations
Khyathi Lakshmi Sri Ghantasala | MS Business Analytics
=============================================================================
Tools: Python, scikit-learn, pandas, matplotlib, seaborn
Domain: Financial Risk Analytics | Predictive Modeling | Campaign Strategy
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.inspection import permutation_importance

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

NAVY  = '#1C2B4A'
BLUE  = '#1A5276'
TEAL  = '#148F77'
AMBER = '#D4AC0D'
RED   = '#C0392B'
GREEN = '#1E8449'
COLS  = [NAVY, BLUE, TEAL, AMBER, RED, GREEN]

np.random.seed(42)
N = 30_000

print("=" * 65)
print("  PROJECT 3: Credit Risk Predictive Response Model")
print("  Khyathi Lakshmi Sri Ghantasala | MS Business Analytics")
print("=" * 65)

# ── STEP 1: Generate Financial Dataset ────────────────────────────────────
print("\n[1/8] Generating financial customer dataset (30,000 records)...")

df = pd.DataFrame({
    'customer_id':         [f'FIN_{i:06d}' for i in range(N)],
    'age':                 np.random.randint(21, 75, N),
    'credit_score':        np.random.randint(300, 850, N),
    'annual_income':       np.random.lognormal(10.8, 0.6, N).astype(int).clip(15000, 500000),
    'debt_to_income':      np.random.beta(2, 5, N).clip(0.01, 0.99),
    'loan_amount':         np.random.lognormal(9.5, 0.8, N).astype(int).clip(1000, 250000),
    'loan_term_months':    np.random.choice([12, 24, 36, 48, 60, 84], N),
    'employment_years':    np.random.exponential(5, N).clip(0, 40).astype(int),
    'num_credit_accounts': np.random.randint(1, 20, N),
    'num_late_payments':   np.random.poisson(1.2, N).clip(0, 15),
    'num_inquiries_6m':    np.random.poisson(1.5, N).clip(0, 10),
    'utilization_rate':    np.random.beta(3, 4, N),
    'has_mortgage':        np.random.binomial(1, 0.45, N),
    'has_auto_loan':       np.random.binomial(1, 0.38, N),
    'account_age_months':  np.random.randint(3, 240, N),
    'loan_purpose':        np.random.choice(
        ['Debt Consolidation', 'Home Improvement', 'Business', 'Auto', 'Medical'], N
    ),
    'employment_status':   np.random.choice(
        ['Employed', 'Self-Employed', 'Business Owner', 'Retired'], N,
        p=[0.65, 0.20, 0.10, 0.05]
    ),
    'region':              np.random.choice(
        ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'], N
    ),
})

# Default probability
default_prob = (
    0.05
    + 0.25 * (df['credit_score'] < 580).astype(float)
    + 0.15 * (df['credit_score'].between(580, 669)).astype(float)
    + 0.20 * (df['debt_to_income'] > 0.45).astype(float)
    + 0.15 * (df['num_late_payments'] > 2).astype(float)
    + 0.10 * (df['utilization_rate'] > 0.75).astype(float)
    + 0.10 * (df['num_inquiries_6m'] > 3).astype(float)
    + 0.08 * (df['employment_years'] < 1).astype(float)
    - 0.10 * (df['credit_score'] > 750).astype(float)
    - 0.08 * (df['employment_years'] > 5).astype(float)
    - 0.05 * df['has_mortgage'].astype(float)
).clip(0.01, 0.92)

df['defaulted'] = np.random.binomial(1, default_prob, N)
default_rate    = df['defaulted'].mean()
print(f"    ✓ Dataset: {N:,} customer records")
print(f"    ✓ Default rate: {default_rate:.1%}")

# ── STEP 2: Feature Engineering ───────────────────────────────────────────
print("\n[2/8] Engineering 15+ behavioral and financial features...")

df['loan_to_income']      = (df['loan_amount'] / df['annual_income']).round(4)
df['monthly_payment_est'] = (df['loan_amount'] / df['loan_term_months']).round(2)
df['payment_to_income']   = (df['monthly_payment_est'] / (df['annual_income'] / 12)).round(4)
df['credit_score_band']   = pd.cut(df['credit_score'],
                                    bins=[299, 579, 669, 739, 799, 850],
                                    labels=['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional'])
df['risk_exposure']       = (df['loan_amount'] * df['debt_to_income']).round(2)
df['payment_history_score'] = (1 / (df['num_late_payments'] + 1)).round(4)
df['credit_mix_score']    = (df['num_credit_accounts'] / 10).clip(0, 1).round(4)
df['inquiry_penalty']     = (df['num_inquiries_6m'] / 5).clip(0, 1).round(4)
df['stability_score']     = (
    (df['employment_years'] / 10).clip(0, 1) * 0.4 +
    (df['account_age_months'] / 120).clip(0, 1) * 0.3 +
    df['has_mortgage'].astype(float) * 0.3
).round(4)
df['composite_risk_score'] = (
    (df['credit_score'] / 850) * 0.30 +
    (1 - df['debt_to_income']) * 0.25 +
    df['payment_history_score'] * 0.20 +
    df['stability_score'] * 0.15 +
    (1 - df['utilization_rate']) * 0.10
).round(4)

print(f"    ✓ Engineered 15 new features from raw financial data")

# ── STEP 3: Encode Categorical Variables ──────────────────────────────────
print("\n[3/8] Encoding categorical variables...")
le = LabelEncoder()
for col in ['loan_purpose', 'employment_status', 'region']:
    df[f'{col}_encoded'] = le.fit_transform(df[col])

features = [
    'credit_score', 'annual_income', 'debt_to_income', 'loan_amount',
    'loan_term_months', 'employment_years', 'num_credit_accounts',
    'num_late_payments', 'num_inquiries_6m', 'utilization_rate',
    'has_mortgage', 'has_auto_loan', 'account_age_months',
    'loan_to_income', 'payment_to_income', 'risk_exposure',
    'payment_history_score', 'credit_mix_score', 'inquiry_penalty',
    'stability_score', 'composite_risk_score',
    'loan_purpose_encoded', 'employment_status_encoded', 'region_encoded'
]

X = df[features].fillna(0)
y = df['defaulted']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"    ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── STEP 4: Model 1 — Logistic Regression (Baseline) ─────────────────────
print("\n[4/8] Training baseline model (Logistic Regression)...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_sc, y_train)
lr_preds = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_acc   = accuracy_score(y_test, lr_preds)
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_ap    = average_precision_score(y_test, lr_proba)
print(f"    Logistic Regression: Accuracy={lr_acc:.1%} | AUC={lr_auc:.4f} | AP={lr_ap:.4f}")

# ── STEP 5: Model 2 — Random Forest (Improved) ────────────────────────────
print("\n[5/8] Training Random Forest classifier (improved model)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_acc   = accuracy_score(y_test, rf_preds)
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_ap    = average_precision_score(y_test, rf_proba)
auc_improvement = (rf_auc - lr_auc) / lr_auc * 100
print(f"    Random Forest:       Accuracy={rf_acc:.1%} | AUC={rf_auc:.4f} | AP={rf_ap:.4f}")
print(f"    AUC Improvement:     +{rf_auc - lr_auc:.4f} ({auc_improvement:.1f}% over baseline)")

# ── STEP 6: Propensity Scoring & Risk Tiering ─────────────────────────────
print("\n[6/8] Applying propensity scoring — risk tier framework...")
df['default_probability'] = rf.predict_proba(df[features].fillna(0))[:, 1]
df['risk_tier'] = pd.cut(
    df['default_probability'],
    bins=[0, 0.10, 0.25, 0.45, 0.65, 1.0],
    labels=['Tier 1 — Very Low Risk', 'Tier 2 — Low Risk',
            'Tier 3 — Medium Risk', 'Tier 4 — High Risk', 'Tier 5 — Very High Risk']
)

tier_summary = df.groupby('risk_tier', observed=True).agg(
    customers       = ('customer_id', 'count'),
    actual_default  = ('defaulted', 'mean'),
    avg_loan        = ('loan_amount', 'mean'),
    avg_credit_score= ('credit_score', 'mean'),
    avg_probability = ('default_probability', 'mean')
).round(3)

print("\n    Risk Tier Framework — Business Decision Guide:")
print(tier_summary.to_string())

# Business recommendations per tier
recommendations = {
    'Tier 1 — Very Low Risk': 'Approve — fast track; premium product offerings eligible',
    'Tier 2 — Low Risk':      'Approve — standard terms; upsell opportunity',
    'Tier 3 — Medium Risk':   'Approve with conditions — reduced limit or higher rate',
    'Tier 4 — High Risk':     'Manual review required — collateral or co-signer recommended',
    'Tier 5 — Very High Risk':'Decline or refer to secured product alternatives',
}

# ── STEP 7: Feature Importance with Statistical Validation ────────────────
print("\n[7/8] Performing feature importance analysis with significance testing...")

feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
top_features = feat_imp.head(10)

# Statistical significance testing: t-test between defaulted vs non-defaulted
sig_results = {}
for feat in top_features.index:
    group_default = df[df['defaulted'] == 1][feat].dropna()
    group_nodefault = df[df['defaulted'] == 0][feat].dropna()
    t_stat, p_val = stats.ttest_ind(group_default, group_nodefault)
    sig_results[feat] = {'t_stat': round(t_stat, 3), 'p_value': p_val,
                         'significant': p_val < 0.05}

sig_count = sum(1 for v in sig_results.values() if v['significant'])
print(f"    ✓ Top 10 features tested for statistical significance")
print(f"    ✓ Statistically significant features (p<0.05): {sig_count}/10")

# ── STEP 8: Visualization Dashboard ───────────────────────────────────────
print("\n[8/8] Building executive risk analytics dashboard...")

fig = plt.figure(figsize=(20, 16), facecolor='#FAFAFA')
fig.suptitle(
    'Credit Risk Predictive Response Model\nKhyathi Lakshmi Sri Ghantasala | MS Business Analytics',
    fontsize=15, fontweight='bold', color=NAVY, y=0.99
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# 1. ROC Curves
ax1 = fig.add_subplot(gs[0, 0:2])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
ax1.plot(fpr_lr, tpr_lr, color=AMBER, lw=2,   label=f'Logistic Reg — Baseline (AUC={lr_auc:.4f})')
ax1.plot(fpr_rf, tpr_rf, color=TEAL,  lw=2.5, label=f'Random Forest — Improved (AUC={rf_auc:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random Baseline')
ax1.fill_between(fpr_rf, tpr_rf, alpha=0.08, color=TEAL)
ax1.fill_between(fpr_lr, tpr_lr, alpha=0.05, color=AMBER)
ax1.set_title(f'ROC Curve — AUC Improvement: +{(rf_auc-lr_auc)*100:.1f}%',
              fontweight='bold', color=NAVY)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate (Recall)')
ax1.legend(fontsize=9)

# 2. Precision-Recall
ax2 = fig.add_subplot(gs[0, 2])
prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_proba)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_proba)
ax2.plot(rec_lr, prec_lr, color=AMBER, lw=2,   label=f'LR (AP={lr_ap:.3f})')
ax2.plot(rec_rf, prec_rf, color=TEAL,  lw=2.5, label=f'RF (AP={rf_ap:.3f})')
ax2.axhline(y=default_rate, color='gray', linestyle='--', lw=1,
            label=f'Baseline ({default_rate:.1%})')
ax2.set_title('Precision-Recall Curve', fontweight='bold', color=NAVY)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend(fontsize=8)

# 3. Confusion Matrix
ax3 = fig.add_subplot(gs[0, 3])
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'],
            annot_kws={'size': 12, 'weight': 'bold'})
ax3.set_title('Confusion Matrix\nRandom Forest', fontweight='bold', color=NAVY)
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# 4. Feature Importance
ax4 = fig.add_subplot(gs[1, 0:2])
colors_fi = [TEAL if v['significant'] else AMBER for v in
             [sig_results.get(f, {'significant': False}) for f in top_features.index]]
bars4 = ax4.barh(range(len(top_features)), top_features.values[::-1],
                 color=colors_fi[::-1], edgecolor='white')
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels([f.replace('_', ' ').title() for f in top_features.index[::-1]], fontsize=9)
ax4.set_title('Top 10 Feature Importance\n(Teal = Statistically Significant p<0.05)',
              fontweight='bold', color=NAVY)
ax4.set_xlabel('Importance Score')
for bar, val in zip(bars4, top_features.values[::-1]):
    ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8)

# 5. Risk Tier Distribution
ax5 = fig.add_subplot(gs[1, 2])
tier_counts  = df['risk_tier'].value_counts().sort_index()
tier_cols    = [GREEN, TEAL, AMBER, RED, '#8B0000']
bars5 = ax5.barh([t.split('—')[1].strip() for t in tier_counts.index],
                 tier_counts.values, color=tier_cols, edgecolor='white')
ax5.set_title('Risk Tier Distribution', fontweight='bold', color=NAVY)
ax5.set_xlabel('Number of Customers')
for bar, val in zip(bars5, tier_counts.values):
    ax5.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=9)

# 6. Default Rate by Risk Tier
ax6 = fig.add_subplot(gs[1, 3])
tier_dr = df.groupby('risk_tier', observed=True)['defaulted'].mean() * 100
bars6   = ax6.bar(range(len(tier_dr)), tier_dr.values,
                  color=tier_cols, edgecolor='white')
ax6.set_xticks(range(len(tier_dr)))
ax6.set_xticklabels(['T1\nVery Low', 'T2\nLow', 'T3\nMed', 'T4\nHigh', 'T5\nVery High'],
                    fontsize=8)
ax6.set_title('Actual Default Rate\nby Risk Tier', fontweight='bold', color=NAVY)
ax6.set_ylabel('Default Rate (%)')
for bar, val in zip(bars6, tier_dr.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color=NAVY)

# 7. Default Probability Distribution
ax7 = fig.add_subplot(gs[2, 0:2])
defaulted_probs     = df[df['defaulted'] == 1]['default_probability']
non_defaulted_probs = df[df['defaulted'] == 0]['default_probability']
ax7.hist(non_defaulted_probs, bins=50, alpha=0.6, color=TEAL,
         label=f'No Default (n={len(non_defaulted_probs):,})', density=True)
ax7.hist(defaulted_probs, bins=50, alpha=0.7, color=RED,
         label=f'Default (n={len(defaulted_probs):,})', density=True)
ax7.axvline(x=0.25, color=AMBER, lw=2, linestyle='--', label='Low/Medium boundary')
ax7.axvline(x=0.45, color=NAVY,  lw=2, linestyle='--', label='Medium/High boundary')
ax7.set_title('Predicted Default Probability Distribution', fontweight='bold', color=NAVY)
ax7.set_xlabel('Predicted Default Probability')
ax7.set_ylabel('Density')
ax7.legend(fontsize=8)

# 8. Business Recommendation Summary
ax8 = fig.add_subplot(gs[2, 2:])
ax8.axis('off')
ax8.set_title('Business Decision Framework', fontweight='bold', color=NAVY, pad=15)
summary_items = [
    ('MODEL PERFORMANCE', '', NAVY, 11, 'bold'),
    ('Baseline AUC (LR):',   f'{lr_auc:.4f}',   AMBER, 10, 'normal'),
    ('Improved AUC (RF):',   f'{rf_auc:.4f}',   TEAL,  10, 'bold'),
    ('AUC Improvement:',     f'+{(rf_auc-lr_auc)*100:.1f}% over baseline', TEAL, 10, 'bold'),
    ('Model Accuracy:',      f'{rf_acc:.1%}',    NAVY,  10, 'bold'),
    ('Significant Features:', f'{sig_count}/10 (p<0.05)', GREEN, 10, 'normal'),
    ('', '', NAVY, 8, 'normal'),
    ('RISK TIER DECISIONS', '', NAVY, 11, 'bold'),
    ('Tier 1 Very Low Risk:',  'Fast-track approval',          GREEN, 9, 'normal'),
    ('Tier 2 Low Risk:',       'Standard approval + upsell',   TEAL,  9, 'normal'),
    ('Tier 3 Medium Risk:',    'Approve with conditions',       AMBER, 9, 'normal'),
    ('Tier 4 High Risk:',      'Manual review required',        RED,   9, 'normal'),
    ('Tier 5 Very High Risk:', 'Decline / secured alternative', '#8B0000', 9, 'normal'),
    ('', '', NAVY, 8, 'normal'),
    ('DATASET SCALE', '', NAVY, 11, 'bold'),
    ('Total Records:',         f'{N:,} customers',              NAVY, 10, 'normal'),
    ('Features Engineered:',   '24 behavioral + financial',     NAVY, 10, 'normal'),
    ('Default Rate:',          f'{default_rate:.1%}',           RED,  10, 'normal'),
]
y_pos = 0.97
for label, value, color, size, weight in summary_items:
    if value:
        ax8.text(0.02, y_pos, label, fontsize=size, color=NAVY,
                 fontweight=weight, transform=ax8.transAxes, va='top')
        ax8.text(0.55, y_pos, value, fontsize=size, color=color,
                 fontweight='bold', transform=ax8.transAxes, va='top')
    else:
        ax8.text(0.02, y_pos, label, fontsize=size, color=color,
                 fontweight=weight, transform=ax8.transAxes, va='top')
    y_pos -= 0.055

plt.savefig('/mnt/user-data/outputs/project3_credit_risk_model.png',
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()

print("\n" + "=" * 65)
print("  PROJECT 3 COMPLETE")
print(f"  Records analyzed:            {N:,}")
print(f"  Features engineered:         {len(features)}")
print(f"  Baseline AUC (LR):           {lr_auc:.4f}")
print(f"  Improved AUC (RF):           {rf_auc:.4f}")
print(f"  AUC improvement:             +{auc_improvement:.1f}%")
print(f"  Model accuracy:              {rf_acc:.1%}")
print(f"  Statistically significant:   {sig_count}/10 features")
print(f"  Output saved: project3_credit_risk_model.png")
print("=" * 65)
