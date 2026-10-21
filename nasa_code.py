# ==================================================================
# 2026 데이터 챌린지 — 타겟별 모델링 재현 코드
#
# 본 코드는 포트폴리오 공개 목적으로, 
# 사내 데이터 챌린지 참가 당시 작성한 재현 코드를 정리한 것입니다. 
# 회사 데이터 보안 정책에 따라 실제 변수 코드, 사내 데이터 사전 파일, 
# 내부 인프라 경로는 모두 익명화·일반화하여 제거했습니다. 
# 로직과 구조 자체는 실제 사용한 방식과 동일합니다.
#
# ==================================================================

import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier


# ==================================================================
# 0단계. 데이터 준비 (공통)
#   Sentinel 처리로 결측을 의미하는 특이값을 결측치로 분리하고
#   (missing_flag 생성), 극소수 특이 패턴(low_card_flag)도 함께 생성한다.
#
# ==================================================================
print("[0] 데이터 준비 중...")
t0 = time.time()

train = pd.read_parquet("./data/train.parquet")
test = pd.read_parquet("./data/test.parquet")
TARGETS_ALL = ["T_A001", "T_B001", "T_C001", "T_D001", "T_E001", "T_F001"]
feature_cols = [c for c in train.columns if c not in TARGETS_ALL + ["ID"]]

COLUMN_ATTR_MAP = {}  # 예: {"feature_01": "금액", "feature_02": "비율", ...}
COLUMN_DESC_MAP = {}  # 예: {"feature_01": "...이용률...", ...}

RATIO_KEYWORDS = ["이용률", "소진율", "증감률", "비율", "경락가율"]
NONNEG_ATTRS = {"건수", "금액", "기관수", "기간-개월수", "기간-일수", "여부"}


def resolve_attr(col):
    base = re.sub(r"_(2[1-4])$", "", col)
    raw = COLUMN_ATTR_MAP.get(base) or COLUMN_ATTR_MAP.get(col)
    desc_text = COLUMN_DESC_MAP.get(base, "") or COLUMN_DESC_MAP.get(col, "")
    if any(kw in str(desc_text) for kw in RATIO_KEYWORDS):
        return "비율(보정)"
    return raw


nonneg_cols = [c for c in feature_cols if resolve_attr(c) in NONNEG_ATTRS]
ratio_cols = [c for c in feature_cols if resolve_attr(c) in ("비율", "비율(보정)")]
code_cols = [c for c in feature_cols if resolve_attr(c) in ("코드", "기타")]


def apply_sentinel(X, cols, group):
    X_new = X.copy()
    flag_dict = {}
    for col in cols:
        if col not in X_new.columns:
            continue
        s = X_new[col]
        mask = pd.Series(False, index=s.index)
        if group == "nonneg":
            mask |= s < 0
            mask |= (s >= 999999990) & (s <= 999999999)
        elif group == "ratio":
            mask |= s == -999
            mask |= s == -99
            mask |= (s >= 999999990) & (s <= 999999999)
        elif group == "code":
            mask |= (s >= 999999990) & (s <= 999999999)
        if mask.any():
            flag_dict[f"{col}_missing_flag"] = mask.astype(int)
            X_new.loc[mask, col] = np.nan
    if flag_dict:
        X_new = pd.concat([X_new, pd.DataFrame(flag_dict, index=X_new.index)], axis=1)
    return X_new


X_base = train[feature_cols].copy()
X_base = apply_sentinel(X_base, nonneg_cols, "nonneg")
X_base = apply_sentinel(X_base, ratio_cols, "ratio")
X_base = apply_sentinel(X_base, code_cols, "code")
X_test_base = test[feature_cols].copy()
X_test_base = apply_sentinel(X_test_base, nonneg_cols, "nonneg")
X_test_base = apply_sentinel(X_test_base, ratio_cols, "ratio")
X_test_base = apply_sentinel(X_test_base, code_cols, "code")
X_base, X_test_base = X_base.align(X_test_base, join="left", axis=1, fill_value=0)

# 특이 패턴(예: 극소수 카드 보유 유형)을 별도 플래그로 추가
LOW_CARD_SENTINELS = [999999991, 999999992, 999999993]


def scan_low_card_values(X, cols):
    result = []
    for col in cols:
        if col not in X.columns:
            continue
        if X[col].isin(LOW_CARD_SENTINELS).sum() > 0:
            result.append({"컬럼": col})
    return pd.DataFrame(result)


all_check_cols = nonneg_cols + ratio_cols + code_cols
low_card_cols = (
    scan_low_card_values(train[feature_cols], all_check_cols)["컬럼"].tolist()
    if all_check_cols
    else []
)


def add_low_card_flag(X_raw, cols):
    mask = pd.Series(False, index=X_raw.index)
    for col in cols:
        if col in X_raw.columns:
            mask |= X_raw[col].isin(LOW_CARD_SENTINELS)
    return mask.astype(int)


X_base["low_card_flag"] = add_low_card_flag(train[feature_cols], low_card_cols)
X_test_base["low_card_flag"] = add_low_card_flag(test[feature_cols], low_card_cols)
print(f"[0] 완료 ({time.time() - t0:.1f}초), shape={X_base.shape}")


# ==================================================================
# 1단계. 공통 함수
#   연도형 파생변수 생성, LGBM/CatBoost 학습, 가중 앙상블 및
#   threshold 탐색 함수. A/D/E/F 전체에서 공통으로 사용한다.
# ==================================================================
def find_best_threshold(y_true, pred):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.01, 0.90, 0.01):
        f1 = f1_score(y_true, (pred >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def find_year_groups_from_top_list(df_columns, top_features):
    """컬럼명이 '..._21'~'..._24' 형태로 연도 suffix를 가질 때,
    같은 베이스 변수의 연도별 컬럼들을 그룹으로 묶는다."""
    year_groups = defaultdict(list)
    top_set = set(top_features)
    for col in df_columns:
        m = re.match(r"(.+)_([2][1-4])$", col)
        if m:
            base = m.group(1)
            if base in top_set or col in top_set:
                year_groups[base].append((int(m.group(2)), col))
    for base in list(year_groups.keys()):
        if base in df_columns:
            year_groups[base].append((25, base))  # 당해년도(suffix 없음)
    for base in year_groups:
        year_groups[base] = sorted(year_groups[base])
    return year_groups


def add_target_year_features_from_top_list(X_input, target, top_features_dict):
    """연도별 추세(mean/std/max/min/diff/growth) 파생변수를 생성한다."""
    X_new = X_input.copy()
    new_features = {}
    top_features = top_features_dict[target]
    groups = find_year_groups_from_top_list(X_new.columns, top_features)
    print(f"  {target} 연도형 그룹 수: {len(groups)}")
    for base, cols in groups.items():
        col_names = [c for _, c in cols]
        if len(col_names) < 2:
            continue
        temp = X_new[col_names]
        new_features[f"{base}_{target}_mean"] = temp.mean(axis=1)
        new_features[f"{base}_{target}_std"] = temp.std(axis=1)
        new_features[f"{base}_{target}_max"] = temp.max(axis=1)
        new_features[f"{base}_{target}_min"] = temp.min(axis=1)
        new_features[f"{base}_{target}_diff"] = temp.iloc[:, -1] - temp.iloc[:, 0]
        new_features[f"{base}_{target}_growth"] = (
            temp.iloc[:, -1] - temp.iloc[:, 0]
        ) / (temp.iloc[:, 0] + 1)
    if new_features:
        X_new = pd.concat([X_new, pd.DataFrame(new_features, index=X_new.index)], axis=1)
    return X_new.loc[:, ~X_new.columns.duplicated()].copy()


def train_lgbm(X_train, y, X_test, n_estimators=2000, lr=0.03, n_splits=5, random_state=42):
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            num_leaves=64,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            force_col_wise=True,
            random_state=random_state,
            n_jobs=-1,
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[early_stopping(100), log_evaluation(0)],
        )
        val_pred = m.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_pred += m.predict_proba(X_test)[:, 1] / n_splits
        print(f"    [LGB] Fold {fold + 1} AP: {average_precision_score(y_val, val_pred):.5f}")
    return oof, test_pred


def train_cat(X_train, y, X_test, n_splits=5, random_state=42):
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=random_state,
            early_stopping_rounds=100,
            verbose=0,
            allow_writing_files=False,
        )
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        val_pred = m.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_pred += m.predict_proba(X_test)[:, 1] / n_splits
        print(f"    [CAT] Fold {fold + 1} AP: {average_precision_score(y_val, val_pred):.5f}")
    return oof, test_pred


def run_ensemble(X_tr, X_te, y, label, n_estimators=2000, lr=0.03):
    lgbm_oof, lgbm_test = train_lgbm(X_tr, y, X_te, n_estimators=n_estimators, lr=lr)
    cat_oof, cat_test = train_cat(X_tr, y, X_te)
    best_score, best_result = -1, None
    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ens_oof = w * lgbm_oof + (1 - w) * cat_oof
        ap = average_precision_score(y, ens_oof)
        threshold, f1 = find_best_threshold(y, ens_oof)
        score = ap * 0.6 + f1 * 0.4
        if score > best_score:
            best_score = score
            best_result = {
                "weight": w,
                "AP": ap,
                "F1": f1,
                "threshold": threshold,
                "score": score,
                "oof": ens_oof,
                "test_pred": w * lgbm_test + (1 - w) * cat_test,
            }
    print(f"  [{label}] AP={best_result['AP']:.5f}, F1={best_result['F1']:.5f}, Score={best_result['score']:.5f}")
    return best_result


results = {}


# ==================================================================
# 2단계. T_A001 (보험)
#   Top-K Gain Importance 변수 + low_card_flag 1개만 추가한 실채점 채택안.
# ==================================================================
print("\n" + "=" * 70 + "\nT_A001 (보험)\n" + "=" * 70)

topk_a = joblib.load("./checkpoints/baseline_importance_T_A001.pkl").head(30)["feature"].tolist()
y_a = train["T_A001"]

cols_a = topk_a + ["low_card_flag"]
results["A"] = run_ensemble(X_base[cols_a], X_test_base[cols_a], y_a, "A: Top-K+flag")


# ==================================================================
# 3단계. T_B001 (주거/구독)
#   ※ (Top-K + flag) 동일하게 적용한 재현 코드
# ==================================================================
print("\n" + "=" * 70 + "\nT_B001 (주거/구독) - 공통 베이스라인 기준\n" + "=" * 70)

topk_b = joblib.load("./checkpoints/baseline_importance_T_B001.pkl").head(30)["feature"].tolist()
y_b = train["T_B001"]

cols_b = topk_b + ["low_card_flag"]
results["B_baseline"] = run_ensemble(X_base[cols_b], X_test_base[cols_b], y_b, "B: Top-K+flag (베이스라인)")


# ==================================================================
# 4단계. T_C001 (교육)
#   ※ (Top-K + flag) 동일하게 적용한 재현 코드
# ==================================================================
print("\n" + "=" * 70 + "\nT_C001 (교육) - 공통 베이스라인 기준\n" + "=" * 70)

topk_c = joblib.load("./checkpoints/baseline_importance_T_C001.pkl").head(30)["feature"].tolist()
y_c = train["T_C001"]

cols_c = topk_c + ["low_card_flag"]
results["C_baseline"] = run_ensemble(X_base[cols_c], X_test_base[cols_c], y_c, "C: Top-K+flag (베이스라인)")


# ==================================================================
# 5단계. T_D001 (여행)
#   Top-K + 결측 플래그 + 연도형 파생변수에, 해외소비 관련 변수의
#   전년 대비 변화량(Delta)을 추가한 조합.
# ==================================================================
print("\n" + "=" * 70 + "\nT_D001 (여행) - 로컬 최선안\n" + "=" * 70)

topk_d = joblib.load("./checkpoints/baseline_importance_T_D001.pkl").head(30)["feature"].tolist()
topk_dict_d = {"T_D001": topk_d}
y_d = train["T_D001"]

X_d = add_target_year_features_from_top_list(X_base, "T_D001", topk_dict_d)
X_d_test = add_target_year_features_from_top_list(X_test_base, "T_D001", topk_dict_d)
X_d, X_d_test = X_d.align(X_d_test, join="left", axis=1, fill_value=0)

# 해외소비 관련 변수의 전년 대비 변화량
# 실제 컬럼 코드는 비공개 처리하고 예시 변수명으로 대체
OVERSEAS_COL, OVERSEAS_COL_PREV = "overseas_spending_feature", "overseas_spending_feature_24"
if all(c in X_d.columns for c in [OVERSEAS_COL, OVERSEAS_COL_PREV]):
    X_d["overseas_delta"] = X_d[OVERSEAS_COL].fillna(0) - X_d[OVERSEAS_COL_PREV].fillna(0)
    X_d_test["overseas_delta"] = X_d_test[OVERSEAS_COL].fillna(0) - X_d_test[OVERSEAS_COL_PREV].fillna(0)

cols_d = X_d.columns.tolist()
results["D_local"] = run_ensemble(
    X_d[cols_d], X_d_test[cols_d], y_d, "D: Top-K+flag+연도형FE+Delta",
    n_estimators=5000, lr=0.02,
)


# ==================================================================
# 6단계. T_E001 (라이프스타일)
#   Top-K 변수 원본 그대로 사용한 실채점 채택안.
# ==================================================================
print("\n" + "=" * 70 + "\nT_E001 (라이프스타일)\n" + "=" * 70)

topk_e = joblib.load("./checkpoints/baseline_importance_T_E001.pkl").head(30)["feature"].tolist()
y_e = train["T_E001"]
results["E"] = run_ensemble(X_base[topk_e], X_test_base[topk_e], y_e, "E: Top-K 원본")


# ==================================================================
# 7단계. T_F001 (모빌리티)
#   Top-K + 연도형 파생변수(mean/std/max/min/diff/growth) 조합.
#   원본 파이프라인 설정(n_estimators=5000, learning_rate=0.02)을 그대로 사용.
# ==================================================================
print("\n" + "=" * 70 + "\nT_F001 (모빌리티)\n" + "=" * 70)

topk_f = joblib.load("./checkpoints/baseline_importance_T_F001.pkl").head(30)["feature"].tolist()
topk_dict_f = {"T_F001": topk_f}
y_f = train["T_F001"]

X_f = add_target_year_features_from_top_list(X_base, "T_F001", topk_dict_f)
X_f_test = add_target_year_features_from_top_list(X_test_base, "T_F001", topk_dict_f)
X_f, X_f_test = X_f.align(X_f_test, join="left", axis=1, fill_value=0)

cols_f = X_f.columns.tolist()
results["F"] = run_ensemble(
    X_f[cols_f], X_f_test[cols_f], y_f, "F: Top-K+연도형FE",
    n_estimators=5000, lr=0.02,
)


print("\n" + "=" * 70 + "\n완료\n" + "=" * 70)
for k, v in results.items():
    print(f"{k}: AP={v['AP']:.5f}, F1={v['F1']:.5f}, Score={v['score']:.5f}")
