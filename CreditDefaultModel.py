import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from watch import Watch
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from data_loader import *
from analyzer import *


def flatten_agg_df_columns(df_agg, prefix=None):
    if prefix is None:
        df_agg.columns = ['_'.join([c0, c1.upper()]) for c0, c1 in df_agg.columns]
    else:
        df_agg.columns = ['_'.join([prefix, c0, c1.upper()]) for c0, c1 in df_agg.columns]
    return df_agg


def append_one_hot_encoding(df, series, prefix=None, dummy_na=True):
    return pd.concat([df, pd.get_dummies(series, prefix=prefix, dummy_na=dummy_na)], axis=1, copy=False)


def group_values(col_orig, new_col_values):
    return pd.DataFrame({col: col_orig.isin(values) for col, values in new_col_values})


def get_preprocessed_bureau_data():
    df_bureau = load_bureau()
    df_bureau_balance = load_bureau_balance()

    df_bureau["CNT_OVERDUE"] = (df_bureau["CREDIT_DAY_OVERDUE"] > 0).astype(np.uint8, copy=False)
    bureau_groups = df_bureau.groupby("SK_ID_CURR")
    df_bureau_agg = flatten_agg_df_columns(bureau_groups.agg({
        "CNT_OVERDUE": ["sum"],
        "AMT_CREDIT_MAX_OVERDUE": ["max"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "sum"],
        "DAYS_CREDIT": ["min", "max"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"]

        # 'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        # 'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        # 'DAYS_CREDIT_UPDATE': ['mean'],
        # 'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        # 'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        # 'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        # 'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        # 'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        # 'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        # 'AMT_ANNUITY': ['max', 'mean'],
        # 'CNT_CREDIT_PROLONG': ['sum'],
        # 'MONTHS_BALANCE_MIN': ['min'],
        # 'MONTHS_BALANCE_MAX': ['max'],
        # 'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }), "BUREAU")
    df_bureau_agg["BUREAU_LOAN_CNT"] = bureau_groups.size()

    # def func_bureau_agg(s):
    #     total_overdues = (s.CREDIT_DAY_OVERDUE > 0).sum()
    #     max_overdue = s.AMT_CREDIT_MAX_OVERDUE.max()
    #     # active_debt = s[s.CREDIT_ACTIVE == "Active"].AMT_CREDIT_SUM_DEBT.sum()
    #     return pd.Series([total_overdues, max_overdue], index=["total_overdues", "max_overdues"])
    # df_bureau_agg = df_bureau.groupby("SK_ID_CURR").apply(func_bureau_agg)

    bureau_active_groups = df_bureau[df_bureau.CREDIT_ACTIVE == "Active"].groupby("SK_ID_CURR")
    df_bureau_active_agg = flatten_agg_df_columns(bureau_active_groups.agg({
        "AMT_CREDIT_SUM_DEBT": ["sum"],
        "DAYS_CREDIT_ENDDATE": ["max"]
    }), "BUREAU_ACTIVE")
    df_bureau_agg = df_bureau_agg.join(df_bureau_active_agg, how="left")

    df_bureau_agg.fillna({
        "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX": 0,
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": 0,
        "BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM": 0,
        # "BUREAU_ACTIVE_DAYS_CREDIT_ENDDATE_MAX": 100000
    }, inplace=True)

    # assert df_bureau_agg.notnull().all(axis=None)

    # func_overdue_ratio = lambda s: (s.STATUS > "0").sum() / len(s)
    # df_bureau_balance.dropna(inplace=True)
    # df_bureau_agg["overdue_month_ratio"] = df_bureau_balance.groupby("SK_ID_BUREAU").apply(func_overdue_ratio)

    del df_bureau, df_bureau_balance
    return df_bureau_agg


def get_preprocessed_previous_app_data(load_inst_pay=True, load_credit_balance=True):
    # def func_prev_app_agg(s):
    #     return pd.Series({"has_assessed_risk": s.has_assessed_risk.sum(),
    #                       "max_refused": s[s.is_refused].AMT_APPLICATION.max(),
    #                       "total_approved": s[s.is_approved].size,
    #                       "max_prev_annuity": s[s.is_approved].AMT_ANNUITY.max()})

    df_prev_app = load_prev_app_data()
    df_prev_app["CNT_ASSESSED_RISK"] = (df_prev_app["NAME_PRODUCT_TYPE"] == "x-sell").astype(np.uint8, copy=False)
    df_prev_app["NAME_YIELD_GROUP_CODE"] = df_prev_app["NAME_YIELD_GROUP"].cat.codes
    prev_app_groups = df_prev_app.groupby("SK_ID_CURR")
    df_prev_app_agg = flatten_agg_df_columns(prev_app_groups.agg({
        "CNT_ASSESSED_RISK": ["sum"],

        # 'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        # 'AMT_ANNUITY': ['min', 'max', 'mean'],
        # 'AMT_APPLICATION': ['min', 'max', 'mean'],
        # 'AMT_CREDIT': ['min', 'max', 'mean'],
        # 'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        # 'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        # 'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        # 'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        # 'DAYS_DECISION': ['min', 'max', 'mean'],
        # 'CNT_PAYMENT': ['mean', 'sum'],
    }), "PREV")

    prev_app_refused_groups = df_prev_app[df_prev_app.NAME_CONTRACT_STATUS == "Refused"].groupby("SK_ID_CURR")
    df_prev_refused_agg = flatten_agg_df_columns(prev_app_refused_groups.agg({
        "AMT_APPLICATION": ["max"],
        "RATE_INTEREST_PRIMARY": ["mean", "max"],
        "NAME_YIELD_GROUP_CODE": ["mean", "max"]
    }), "PREV_REFUSED")
    df_prev_refused_agg["PREV_REFUSED_CNT"] = prev_app_refused_groups.size()
    df_prev_app_agg = df_prev_app_agg.join(df_prev_refused_agg, how="left")
    del df_prev_refused_agg

    prev_app_approved_groups = df_prev_app[df_prev_app.NAME_CONTRACT_STATUS == "Approved"].groupby("SK_ID_CURR")
    df_prev_approved_agg = flatten_agg_df_columns(prev_app_approved_groups.agg({
        "AMT_APPLICATION": ["max"],
        "AMT_ANNUITY": ["max"],
        "RATE_INTEREST_PRIMARY": ["mean", "max"],
        "NAME_YIELD_GROUP_CODE": ["mean", "max"]
    }), "PREV_APPROVED")
    df_prev_approved_agg["PREV_APPROVED_CNT"] = prev_app_approved_groups.size()
    df_prev_app_agg = df_prev_app_agg.join(df_prev_approved_agg, how="left")
    del df_prev_approved_agg
    del df_prev_app

    if load_inst_pay:
        df_inst_pay = load_install_payments()
        # df_POS_CASH_balance = load_pos_balance()

        # df_prev_app_processed = df_prev_app[["SK_ID_CURR"]].copy()
        # df_prev_app_processed[""] = df_prev_app["NAME_CONTRACT_TYPE"]

        df_inst_pay["CNT_OVERDUE"] = (df_inst_pay.AMT_OVERDUE > 0)
        df_inst_pay["CNT_DPD30"] = (df_inst_pay.AMT_DPD30 > 0)
        df_inst_pay_groups = df_inst_pay.groupby("SK_ID_CURR")

        df_inst_pay_agg = flatten_agg_df_columns(df_inst_pay_groups.agg({
            "AMT_OVERDUE": ["max"],
            "CNT_OVERDUE": ["sum", "mean"],
            "AMT_DPD30": ["max"],
            "CNT_DPD30": ["sum", "mean"],
            "AMT_UNPAID": ["sum"]
        }), "INST_PAY")

        # print_null_columns(df_inst_pay_agg)
        df_prev_app_agg = df_prev_app_agg.join(df_inst_pay_agg, how="outer")
        del df_inst_pay,  df_inst_pay_agg

    if load_credit_balance:
        df_credit_card_balance = load_credit_balance()

        df_credit_card_balance["CNT_OVERDUE"] = (df_credit_card_balance["SK_DPD"] > 0).astype(np.uint8)
        df_credit_card_balance["CNT_OVERDUE_DEF"] = (df_credit_card_balance["SK_DPD_DEF"] > 0).astype(np.uint8)
        cc_groups = df_credit_card_balance.groupby("SK_ID_CURR")
        df_cc_agg = flatten_agg_df_columns(cc_groups.agg({
            "MONTHS_BALANCE": ["min", "max"],
            "SK_DPD": ["max"],
            "SK_DPD_DEF": ["max"],
            "CNT_OVERDUE": ["sum"],
            "CNT_OVERDUE_DEF": ["sum"]
        }), "CREDIT_CARD")
        df_prev_app_agg = df_prev_app_agg.join(df_cc_agg, how="outer")
        del df_credit_card_balance, df_cc_agg

    df_prev_app_agg.fillna({
        "PREV_REFUSED_AMT_APPLICATION_MAX": 0,
        "PREV_REFUSED_CNT": 0,
        "PREV_APPROVED_AMT_APPLICATION_MAX": 0,
        "PREV_APPROVED_AMT_ANNUITY_MAX": 0,
        "PREV_APPROVED_CNT": 0,
        "PREV_CNT_ASSESSED_RISK_SUM": 0,
        "INST_PAY_AMT_OVERDUE_MAX": 0,
        "INST_PAY_CNT_OVERDUE_SUM": 0,
        "INST_PAY_CNT_OVERDUE_MEAN": 0,
        "INST_PAY_AMT_DPD30_MAX": 0,
        "INST_PAY_CNT_DPD30_SUM": 0,
        "INST_PAY_CNT_DPD30_MEAN": 0,
        "INST_PAY_AMT_UNPAID_SUM": 0,
        "CREDIT_CARD_MONTHS_BALANCE_MAX": 0,
        "CREDIT_CARD_MONTHS_BALANCE_MIN": 0,
        "CREDIT_CARD_SK_DPD_MAX": 0,
        "CREDIT_CARD_SK_DPD_DEF_MAX": 0,
        "CREDIT_CARD_CNT_OVERDUE_SUM": 0,
        "CREDIT_CARD_CNT_OVERDUE_DEF_SUM": 0
    }, inplace=True)

    # print(df_prev_app_agg.head())
    # print_null_columns(df_prev_app_agg)
    return df_prev_app_agg


def preprocess_app(df, df_bureau_agg, df_prev_app_agg):
    perform_grouping = True
    excluded_columns = ["TARGET"]
    unchanged_columns = ["CNT_CHILDREN", "CNT_FAM_MEMBERS",
                         "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                         "REGION_POPULATION_RELATIVE",
                         "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
                         "OWN_CAR_AGE",
                         "FLAG_EMP_PHONE",
                         # "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                         # "FLAG_EMAIL",
                         "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
                         # "HOUR_APPR_PROCESS_START",
                         # "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                         # "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
                         "DAYS_LAST_PHONE_CHANGE",
                         "TOTALAREA_MODE", "AMT_REQ_CREDIT_BUREAU_MON"]
    # unchanged_columns.extend(("FLAG_DOCUMENT_" + str(i)) for i in range(2, 22))

    missing_fill_fix_val = {"AMT_GOODS_PRICE": 0, "AMT_ANNUITY": 0, "OWN_CAR_AGE": -1,
                            "OBS_30_CNT_SOCIAL_CIRCLE": 0, "DEF_30_CNT_SOCIAL_CIRCLE": 0,
                            "OBS_60_CNT_SOCIAL_CIRCLE": 0, "DEF_60_CNT_SOCIAL_CIRCLE": 0
                            }

    housing_columns = ["APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
                       "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
                       "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",
                       "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
                       "NONLIVINGAREA_AVG",
                       "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
                       "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE",
                       "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE",
                       "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE",
                       "NONLIVINGAREA_MODE",
                       "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",
                       "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",
                       "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
                       "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
                       # 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
                       ]

    df_app_processed = df[[]].copy()
    df_app_processed["NAME_CONTRACT_TYPE"] = df["NAME_CONTRACT_TYPE"].str.startswith("C").astype(np.uint8)
    df_app_processed["IS_MALE"] = df["CODE_GENDER"].cat.codes
    df_app_processed["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].cat.codes
    df_app_processed["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].cat.codes
    if perform_grouping:
        name_type_suite_groups = [("Acc_No", ["Unaccompanied"]),
                                  ("Acc_Fam_Ch", ["Family", "Children", "Group of people"]),
                                  ("Acc_Spouse", ["Spouse, partner"]),
                                  ("Acc_Other", ["Other_A", "Other_B"])]
        df_app_processed = pd.concat(
            [df_app_processed, group_values(df["NAME_TYPE_SUITE"], name_type_suite_groups)], axis=1, copy=False)

        family_status_groups = [("With_family", ["Married", "Civil marriage"]),
                                ("Without_family", ["Single / not married", "Separated", "Widow"])]
        df_app_processed = pd.concat(
            [df_app_processed, group_values(df["NAME_FAMILY_STATUS"], family_status_groups)], axis=1, copy=False)

        income_type_groups = [("Income_Job", ["Working", "Maternity leave"]),
                              ("Income_Commercial", ["Commercial associate", "Businessman"]),
                              ("Income_Pensioner", ["Pensioner"]),
                              ("Income_Servant", ["State servant"]),
                              ("Income_Other", ["Unemployed", "Student"])]
        df_app_processed = pd.concat(
            [df_app_processed, group_values(df["NAME_INCOME_TYPE"], income_type_groups)], axis=1, copy=False)

        organization_groups = [("Org_Missing", [np.nan]),
                               ("Org_Business_1", ["Business Entity Type 1"]),
                               ("Org_Business_2", ["Business Entity Type 2"]),
                               ("Org_Business_3", ["Business Entity Type 3"]),
                               ("Org_Government", ["Government"]),
                               ("Org_Self", ["Self-employed"]),
                               ("Org_Trade_7", ["Trade: type 7"]),
                               ("Org_Transport_3", ["Transport: type 3"]),
                               ("Org_Transport_4", ["Transport: type 4"]),
                               ("Org_Medicine", ["Medicine"]),
                               ("Org_Other", ["Other"]),
                               ("Org_Mix_0", ["Trade: type 6", "Transport: type 1", "Industry: type 12"]),
                               ("Org_Mix_1", ["Bank", "Military", "Police", "University", "Security Ministries"]),
                               ("Org_Mix_2", ["School", "Insurance", "Culture"]),
                               ("Org_Mix_3", ["Trade: type 5", "Trade: type 4", "Religion"]),
                               ("Org_Mix_4", ["Hotel", "Industry: type 10", "Medicine"]),
                               ("Org_Mix_5", ["Industry: type 3", "Realtor", "Agriculture",
                                              "Trade: type 3", "Industry: type 4", "Security"]),
                               ("Org_Mix_6", ["Industry: type 11", "Postal"]),
                               ("Org_Mix_7", ["Industry: type 13", "Industry: type 8", "Restaurant",
                                              "Construction", "Cleaning", "Industry: type 1"]),
                               ]
        df_app_processed = pd.concat(
            [df_app_processed, group_values(df["ORGANIZATION_TYPE"], organization_groups)], axis=1, copy=False)

        housing_groups = [
                          ("Housing_Missing", [np.nan]),
                          ("Housing_Own", ["House / apartment"]),
                          ("Housing_Provided", ["Municipal apartment", "Office apartment", "Co-op apartment"]),
                          ("Housing_Rent", ["Rented apartment"]),
                          ("Housing_Parent", ["With parents"])
                         ]
        df_app_processed = pd.concat(
            [df_app_processed, group_values(df["NAME_HOUSING_TYPE"], housing_groups)], axis=1, copy=False)
    else:
        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_TYPE_SUITE"], prefix="Acc")
        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_FAMILY_STATUS"], prefix="Fam")
        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_INCOME_TYPE"], prefix="Income")
        df_app_processed = append_one_hot_encoding(df_app_processed, df["ORGANIZATION_TYPE"], prefix="Org")
        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_HOUSING_TYPE"], prefix="Housing")

    df_app_processed["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].cat.codes
    df_app_processed[unchanged_columns] = df[unchanged_columns]
    df_app_processed["EXT_SCORE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df_app_processed["ANN_PERCENT"] = df.AMT_ANNUITY / df.AMT_INCOME_TOTAL
    df_app_processed["EMPLOYED_PERCENT"] = df.DAYS_EMPLOYED / df.DAYS_BIRTH
    df_app_processed["PAYMENT_DURATION"] = df.AMT_CREDIT / df.AMT_ANNUITY
    # df_app_processed["LEFT_OVER"] = df.AMT_INCOME_TOTAL - df.AMT_ANNUITY
    for col, fill_val in missing_fill_fix_val.items():
        df_app_processed[col] = df[col].fillna(fill_val)
        # df_app_processed[col] = df[col]
    # df_app_processed[housing_columns] = df[housing_columns]

    if df_bureau_agg is not None:
        df_app_processed = df_app_processed.join(df_bureau_agg, how="left")
    if df_prev_app_agg is not None:
        df_app_processed = df_app_processed.join(df_prev_app_agg, how="left")
        df_app_processed['ANNUITY_RATIO'] = df_app_processed.AMT_ANNUITY / df_app_processed.PREV_APPROVED_AMT_ANNUITY_MAX

    # for name, col in df.iteritems():
    #     if not (col.isnull().any() or col.dtype == "object" or name in excluded_columns):
    #         # df_app_processed[name] = col
    #         print(name)
    # for col in unchanged_columns:
    #     df_app_processed[col] = df[col]

    return df_app_processed


def run():
    cross_validation = True
    perform_imputation = False
    stop_after_validation = True
    rand_seed = 1

    print("Reading data")
    read_watch = Watch("Reading data")
    read_watch.start()
    df_app_train, df_app_test = load_app_data()
    read_watch.stop()
    print("Finish reading data")

    missing_fill_mean = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    missing_fill_most_freq = ["CNT_FAM_MEMBERS", "AMT_ANNUITY", "DAYS_LAST_PHONE_CHANGE"]

    mean_imputer = SimpleImputer(strategy="mean")
    most_freq_imputer = SimpleImputer(strategy="most_frequent")

    preprocess_watch = Watch("Preprocess")
    print("Preprocess training data")
    preprocess_watch.start()

    df_bureau_agg = None
    df_prev_app_agg = None
    df_bureau_agg = get_preprocessed_bureau_data()
    print("Finish preprocessing bureau data")
    # df_prev_app_agg = get_preprocessed_previous_app_data(False, False)
    # print("Finish preprocessing previous application data")

    df_app_train = shuffle(df_app_train, random_state=rand_seed)
    X_train = preprocess_app(df_app_train, df_bureau_agg, df_prev_app_agg)
    if perform_imputation:
        X_train[missing_fill_mean] = pd.DataFrame(mean_imputer.fit_transform(df_app_train[missing_fill_mean]),
                                                  index=df_app_train.index)
        X_train[missing_fill_most_freq] = pd.DataFrame(most_freq_imputer.fit_transform(df_app_train[missing_fill_most_freq]),
                                                       index=df_app_train.index)
    else:
        X_train[missing_fill_mean] = df_app_train[missing_fill_mean]
        X_train[missing_fill_most_freq] = df_app_train[missing_fill_most_freq]
    y_train = df_app_train["TARGET"]

    print("Preprocess test data")
    X_test = preprocess_app(df_app_test, df_bureau_agg, df_prev_app_agg)
    if perform_imputation:
        X_test[missing_fill_mean] = pd.DataFrame(mean_imputer.transform(df_app_test[missing_fill_mean]),
                                                 index=df_app_test.index)
        X_test[missing_fill_most_freq] = pd.DataFrame(most_freq_imputer.transform(df_app_test[missing_fill_most_freq]),
                                                      index=df_app_test.index)
    else:
        X_test[missing_fill_mean] = df_app_test[missing_fill_mean]
        X_test[missing_fill_most_freq] = df_app_test[missing_fill_most_freq]

    if not X_test.columns.equals(X_train.columns):
        X_test[X_train.columns.difference(X_test.columns)] = 0
        X_test.drop(X_test.columns.difference(X_train.columns), axis=1, inplace=True)
        X_test = X_test.reindex(columns=X_train.columns, axis=1)
    assert X_train.columns.equals(X_test.columns)

    preprocess_watch.stop()

    print("Training data shape:", X_train.shape)
    X_train.info(verbose=5)

    print("Initializing classifier")
    weight_dict = {0: 1, 1: 1}
    clf = XGBClassifier(max_depth=10, min_child_weight=10, seed=rand_seed, tree_method="gpu_hist")
    # clf = XGBClassifier(max_depth=8, min_child_weight=12, seed=1)
    # clf = GradientBoostingClassifier(max_depth=10, min_samples_split=15, verbose=5)
    # clf = DecisionTreeClassifier(class_weight=weight_dict, max_depth=15, min_samples_split=4)
    # clf = LogisticRegression(class_weight=weight_dict)

    # clf = LGBMClassifier(
    #     n_jobs=8,
    #     n_estimators=10000,
    #     learning_rate=0.02,
    #     num_leaves=34,
    #     colsample_bytree=0.9497036,
    #     subsample=0.8715623,
    #     max_depth=8,
    #     reg_alpha=0.041545473,
    #     reg_lambda=0.0735294,
    #     min_split_gain=0.0222415,
    #     min_child_weight=39.3259775,
    #     silent=-1,
    #     verbose=-1)

    print("Choosing classifier parameters")
    # model_selection_watch = Watch("Model selection")
    # params = {"max_depth": [5, 8, 10], "min_child_weight": [10, 12]}
    # model_selection_watch.start()
    # grid_clf = GridSearchCV(clf, param_grid=params, scoring="roc_auc", cv=5, verbose=5).fit(X_train, y_train)
    # model_selection_watch.stop()
    # print(grid_clf.best_score_)
    # print(grid_clf.best_params_)
    # print(grid_clf.cv_results_)
    # clf = grid_clf.best_estimator_
    w = Watch("Validation")
    w.start()
    if cross_validation:
        k_fold = 5
        print("Perform {:d}-fold cross validation".format(k_fold))
        score_val = sum(cross_val_score(clf, X_train, y_train,
                                        cv=k_fold, scoring="roc_auc", verbose=5, n_jobs=2)) / k_fold
    else:
        test_size = 0.1
        print("Perform hold-out validation (Test size: {:.0%})".format(test_size))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=rand_seed)
        print(X_train[:10])
        clf.fit(X_train, y_train)
        # mean_imputer.transform(X_val[missing_fill_mean])
        # most_freq_imputer.transform(X_val[missing_fill_most_freq])
        prob_val = clf.predict_proba(X_val)[:, 1]
        score_val = roc_auc_score(y_val, prob_val)
    w.stop()
    print("Validation AUC: %.6f" % score_val)
    # print(clf.feature_importances_)

    if stop_after_validation:
        Watch.print_all()
        return

    print("Training classifier")
    train_watch = Watch("Training")
    train_watch.start()
    clf.fit(X_train, y_train)
    train_watch.stop()

    print("Dumping trained classifier")
    from joblib import dump
    dump(clf, 'boost_tree_gpu_0.joblib')

    print("Classify test set")
    train_prob_df = pd.DataFrame(clf.predict_proba(X_train)[:, 1], index=X_train.index, columns=["PRED_PROB"])
    train_prob_df.to_csv("train_prob.csv")
    test_prob_df = pd.DataFrame(clf.predict_proba(X_test)[:, 1], index=X_test.index, columns=["TARGET"])
    test_prob_df.to_csv("submission.csv")

    Watch.print_all()


if __name__ == "__main__":
    run()
