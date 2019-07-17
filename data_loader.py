import pandas as pd
import numpy as np
from watch import Watch
import gc
from pandas.api.types import CategoricalDtype

LoanType = CategoricalDtype(["Cash loans", "Revolving loans", "Consumer loans"], False)
HouseType = CategoricalDtype(["block of flats", "terraced house", "specific housing"])
WeekDayType = CategoricalDtype(['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY'], True)
YesNoType = CategoricalDtype(["N", "Y"], True)
LoanStatusType = CategoricalDtype(['Approved', 'Refused', 'Canceled', 'Unused offer'], False)
EducationType = CategoricalDtype(["Incomplete higher", "Higher education", "Lower secondary",
                                  "Secondary / secondary special", "Academic degree"], True)
FamilyType = CategoricalDtype(["Civil marriage", "Married", "Separated", "Single / not married", "Widow"], False)
HousingType = CategoricalDtype(["House / apartment", "Rented apartment", "With parents",
                                "Municipal apartment", "Office apartment", "Co-op apartment"], False)
IncomeType = CategoricalDtype(["Unemployed", "Student", "State servant", "Working", "Commercial associate",
                               "Businessman", "Maternity leave", "Pensioner"], False)
AccompanyType = CategoricalDtype(["Unaccompanied", "Spouse", "partner", "Family", "Children",
                                  "Group of people", "Other_A", "Other_B"], False)
OccupationType = CategoricalDtype(["Accountants", "Cleaning staff", "Cooking staff", "Core staff",
                                   "Drivers", "HR staff", "High skill tech staff", "IT staff", "Laborers",
                                   "Low-skill Laborers", "Managers", "Medicine staff",
                                   "Private service staff", "Realty agents", "Sales staff", "Secretaries",
                                   "Security staff", "Waiters/barmen staff"], False)
CreditStatusType = CategoricalDtype(["Closed", "Active", "Sold", "Bad debt", "Signed"], True)


def load_app_data(train_only=False):
    gender_type = CategoricalDtype(["M", "F"], False)
    yes_no_type2 = CategoricalDtype(["No", "Yes"], True)
    wall_material_type = CategoricalDtype(["Stone, brick", "Wooden", "Block", "Panel", "Monolithic", "Mixed",
                                           "Others"], False)
    fondkapremont_type = CategoricalDtype(['reg oper account', 'org spec account',
                                           'reg oper spec account', 'not specified'], False)

    col_types = {
        "SK_ID_CURR": np.uint32, "TARGET": np.bool, "CODE_GENDER": gender_type, "NAME_CONTRACT_TYPE": LoanType,
        "FLAG_OWN_CAR": YesNoType, "FLAG_OWN_REALTY": YesNoType, "CNT_CHILDREN": np.uint8,
        "AMT_INCOME_TOTAL": np.float32, "AMT_CREDIT": np.float32,
        "AMT_ANNUITY": np.float32, "AMT_GOODS_PRICE": np.float32,
        "NAME_TYPE_SUITE": AccompanyType, "NAME_EDUCATION_TYPE": EducationType, "NAME_INCOME_TYPE": IncomeType,
        "NAME_FAMILY_STATUS": FamilyType, "NAME_HOUSING_TYPE": HousingType,
        "REGION_POPULATION_RELATIVE": np.float32,
        "REGION_RATING_CLIENT": np.uint8, "REGION_RATING_CLIENT_W_CITY": np.uint8,
        "WEEKDAY_APPR_PROCESS_START": WeekDayType, "HOUR_APPR_PROCESS_START": np.uint8,
        "DAYS_EMPLOYED": np.float32, "DAYS_BIRTH": np.int32,
        "DAYS_REGISTRATION": np.float32, "DAYS_ID_PUBLISH": np.float32,
        "OWN_CAR_AGE": np.float16,
        "EXT_SOURCE_1": np.float32, "EXT_SOURCE_2": np.float32, "EXT_SOURCE_3": np.float32,
        "FLAG_MOBIL": np.bool, "FLAG_EMP_PHONE": np.bool, "FLAG_WORK_PHONE": np.bool,
        "FLAG_CONT_MOBILE": np.bool, "FLAG_PHONE": np.bool, "FLAG_EMAIL": np.bool,
        "OCCUPATION_TYPE": OccupationType, "CNT_FAM_MEMBERS": np.float16,
        "REG_REGION_NOT_LIVE_REGION": np.bool, "REG_REGION_NOT_WORK_REGION": np.bool,
        "LIVE_REGION_NOT_WORK_REGION": np.bool, "REG_CITY_NOT_LIVE_CITY": np.bool,
        "REG_CITY_NOT_WORK_CITY": np.bool, "LIVE_CITY_NOT_WORK_CITY": np.bool,
        "ORGANIZATION_TYPE": "category",
        "OBS_30_CNT_SOCIAL_CIRCLE": np.float16, "DEF_30_CNT_SOCIAL_CIRCLE": np.float16,
        "OBS_60_CNT_SOCIAL_CIRCLE": np.float16, "DEF_60_CNT_SOCIAL_CIRCLE": np.float16,
        "DAYS_LAST_PHONE_CHANGE": np.float32,
        "AMT_REQ_CREDIT_BUREAU_HOUR": np.float16, "AMT_REQ_CREDIT_BUREAU_DAY": np.float16,
        "AMT_REQ_CREDIT_BUREAU_WEEK": np.float16, "AMT_REQ_CREDIT_BUREAU_MON": np.float16,
        "AMT_REQ_CREDIT_BUREAU_QRT": np.float16, "AMT_REQ_CREDIT_BUREAU_YEAR": np.float16,
        "HOUSETYPE_MODE": HouseType, "FONDKAPREMONT_MODE": fondkapremont_type,
        "WALLSMATERIAL_MODE": wall_material_type, "EMERGENCYSTATE_MODE": yes_no_type2
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
                       "NONLIVINGAREA_MODE", "TOTALAREA_MODE",
                       "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",
                       "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",
                       "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
                       "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
                       # 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
                       ]

    col_types.update(("FLAG_DOCUMENT_" + str(v), np.bool) for v in range(2, 22))
    col_types.update((col, np.float16) for col in housing_columns)
    replace_dict = {"DAYS_EMPLOYED": {365243: np.nan}}

    na_values = ["XNA", "Unknown"]
    df_app_train = pd.read_csv(r"data\application_train.csv", index_col=0, na_values=na_values, dtype=col_types)
    df_app_train.replace(replace_dict, inplace=True)
    assert df_app_train.shape == (307511, 121)
    if train_only:
        return df_app_train
    else:
        df_app_test = pd.read_csv(r"data\application_test.csv", index_col=0, na_values=na_values, dtype=col_types)
        df_app_test.replace(replace_dict, inplace=True)
        assert df_app_test.shape == (48744, 120)
        return df_app_train, df_app_test


def load_prev_app_data():
    interest_types = CategoricalDtype(["low_action", "low_normal", "middle", "high"], True)
    col_types = {
        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "NAME_CONTRACT_TYPE": LoanType,
        "AMT_ANNUITY": np.float32, "AMT_APPLICATION": np.float32, "AMT_CREDIT": np.float32,
        "AMT_DOWN_PAYMENT": np.float32, "AMT_GOODS_PRICE": np.float32,
        "WEEKDAY_APPR_PROCESS_START": WeekDayType, "HOUR_APPR_PROCESS_START": np.uint8,
        "FLAG_LAST_APPL_PER_CONTRACT": YesNoType, "NFLAG_LAST_APPL_IN_DAY": np.bool,
        "RATE_DOWN_PAYMENT": np.float32, "RATE_INTEREST_PRIMARY": np.float32, "RATE_INTEREST_PRIVILEGED": np.float32,
        "NAME_CASH_LOAN_PURPOSE": "category", "NAME_CONTRACT_STATUS": LoanStatusType,
        "DAYS_DECISION": np.int16, "NAME_PAYMENT_TYPE": "category", "CODE_REJECT_REASON": "category",
        "NAME_TYPE_SUITE": "category", "NAME_CLIENT_TYPE": "category", "NAME_GOODS_CATEGORY": "category",
        "NAME_PORTFOLIO": "category", "NAME_PRODUCT_TYPE": "category", "CHANNEL_TYPE": "category",
        "SELLERPLACE_AREA": np.float32, "NAME_SELLER_INDUSTRY": "category", "CNT_PAYMENT": np.float16,
        "NAME_YIELD_GROUP": interest_types, "PRODUCT_COMBINATION": "category",
        "DAYS_FIRST_DRAWING": np.float32, "DAYS_FIRST_DUE": np.float32,
        "DAYS_LAST_DUE_1ST_VERSION": np.float32, "DAYS_LAST_DUE": np.float32, "DAYS_TERMINATION": np.float32,
        "NFLAG_INSURED_ON_APPROVAL": np.float16
    }

    d = {365243: np.nan}

    replace_dict = {"SELLERPLACE_AREA": {-1: np.nan},
                    "DAYS_FIRST_DRAWING": d, "DAYS_FIRST_DUE": d,
                    "DAYS_LAST_DUE_1ST_VERSION": d, "DAYS_LAST_DUE": d, "DAYS_TERMINATION": d}
    df_prev_app = pd.read_csv(r"data\previous_application.csv", na_values=["XNA", ""], index_col=0,
                              keep_default_na=False, dtype=col_types)
    # df_prev_app.set_index("SK_ID_PREV")
    df_prev_app.replace(replace_dict, inplace=True)
    assert df_prev_app.shape == (1670214, 36), "Incorrect shape for df_prev_app: {}".format(df_prev_app.shape)
    return df_prev_app


def load_install_payments(load_processed=True):
    col_types = {
        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32,
        "NUM_INSTALMENT_VERSION": np.uint8, "NUM_INSTALMENT_NUMBER": np.uint16,
        "DAYS_INSTALMENT": np.int16, "DAYS_ENTRY_PAYMENT": np.float16,
        "AMT_INSTALMENT": np.float32, "AMT_PAYMENT": np.float32,
        "NUM_PAYMENTS": np.uint16, "AMT_OVERDUE": np.float32, "AMT_DPD30": np.float32
    }

    if load_processed:
        return pd.read_csv(r"data\installments_payments_processed.csv", dtype=col_types)
    else:
        df = pd.read_csv(r"data\installments_payments.csv", dtype=col_types)
        assert df.shape == (13605401, 8)
        return df


def load_pos_balance():
    col_types = {
        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "MONTHS_BALANCE": np.int16,
        "CNT_INSTALMENT": np.float16, "CNT_INSTALMENT_FUTURE": np.float16,
        "NAME_CONTRACT_STATUS": CreditStatusType,
        # "NAME_CONTRACT_STATUS": "category",
        "SK_DPD": np.uint16, "SK_DPD_DEF": np.uint16
    }
    df = pd.read_csv(r"data\POS_CASH_balance.csv", na_values=["XNA"], dtype=col_types)
    assert df.shape == (10001358, 8)
    return df


def load_credit_balance():
    col_types = {
        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "MONTHS_BALANCE": np.int16,
        "AMT_BALANCE": np.float32, "AMT_CREDIT_LIMIT_ACTUAL": np.float32,
        "AMT_DRAWINGS_ATM_CURRENT": np.float32, "AMT_DRAWINGS_CURRENT": np.float32,
        "AMT_DRAWINGS_OTHER_CURRENT": np.float32, "AMT_DRAWINGS_POS_CURRENT": np.float32,
        "AMT_INST_MIN_REGULARITY": np.float32, "AMT_PAYMENT_CURRENT": np.float32,
        "AMT_PAYMENT_TOTAL_CURRENT": np.float32, "AMT_RECEIVABLE_PRINCIPAL": np.float32,
        "AMT_RECIVABLE": np.float32, "AMT_TOTAL_RECEIVABLE": np.float32,
        "CNT_DRAWINGS_ATM_CURRENT": np.float16, "CNT_DRAWINGS_CURRENT": np.float16,
        "CNT_DRAWINGS_OTHER_CURRENT": np.float16, "CNT_DRAWINGS_POS_CURRENT": np.float16,
        "CNT_INSTALMENT_MATURE_CUM": np.float16, "NAME_CONTRACT_STATUS": "category",
        "SK_DPD": np.uint16, "SK_DPD_DEF": np.uint16
    }
    df = pd.read_csv(r"data\credit_card_balance.csv", dtype=col_types)
    assert df.shape == (3840312, 23)
    return df


def load_bureau():
    currency_types = CategoricalDtype(["currency " + str(v) for v in range(1, 5)], False)
    col_types = {
        "SK_ID_BUREAU": np.uint32, "SK_ID_CURR": np.uint32, "CREDIT_ACTIVE": CreditStatusType,
        "CREDIT_CURRENCY": currency_types, "DAYS_CREDIT": np.float16, "CREDIT_DAY_OVERDUE": np.uint16,
        "DAYS_ENDDATE_FACT": np.float16, "DAYS_CREDIT_ENDDATE": np.float16,
        "AMT_CREDIT_MAX_OVERDUE": np.float32, "CNT_CREDIT_PROLONG": np.uint8,
        "AMT_CREDIT_SUM": np.float32, "AMT_CREDIT_SUM_DEBT": np.float32, "AMT_CREDIT_SUM_LIMIT": np.float32,
        "AMT_CREDIT_SUM_OVERDUE": np.float32, "AMT_ANNUITY": np.float32,
        "CREDIT_TYPE": "category"
    }
    df = pd.read_csv(r"data\bureau.csv", index_col=1, dtype=col_types)
    assert df.shape == (1716428, 16)
    return df


BureauBalanceStatusType = CategoricalDtype(["C"] + list(map(str, range(6))), True)


def load_bureau_balance():
    # status_types = CategoricalDtype(["C"] + list(map(str, range(6))), True)
    col_types = {
        "SK_ID_BUREAU": np.uint32, "MONTHS_BALANCE": np.int16, "STATUS": BureauBalanceStatusType
    }
    df = pd.read_csv(r"data\bureau_balance.csv", na_values=["X"], dtype=col_types)
    assert df.shape == (27299925, 3)
    return df


# def preprocess_prev_app(df):

def print_memory_usage(df, name, show_columns=False):
    usage = df.memory_usage(deep=True) / (1024 * 1024)
    print("Dataframe {0:s} memory usage: {1:.2f} MB".format(name, usage.sum()))
    if show_columns:
        print(usage)


def clean_inst_pay():
    df_inst_pay = load_install_payments(False)
    print_memory_usage(df_inst_pay, "installment_payments")

    df_inst_pay.DAYS_ENTRY_PAYMENT.fillna(0, inplace=True)
    df_inst_pay.AMT_PAYMENT.fillna(-1, inplace=True)
    df_inst_pay_valid_filter = (df_inst_pay["AMT_PAYMENT"] > 0) | (df_inst_pay["AMT_INSTALMENT"] > 0)
    print("Remove {:d} invalid records.".format((~df_inst_pay_valid_filter).sum()))
    df_inst_pay_group = df_inst_pay[df_inst_pay_valid_filter].groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER",
                                                                       "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"])
    del df_inst_pay_valid_filter

    w = Watch("Aggregation 1")
    print("Aggregate multiple installments for one payment")
    w.start()
    df_inst_pay_group_cnt = df_inst_pay_group.size()
    df_inst_agg = df_inst_pay_group.agg({
        "SK_ID_CURR": ["min", "max"],
        "NUM_INSTALMENT_VERSION": ["max", "nunique"],
        "DAYS_INSTALMENT": ["min", "max"],
        "AMT_INSTALMENT": ["min", "max", "sum"]
    })
    df_inst_agg.columns = ['_'.join(col) for col in df_inst_agg.columns]
    del df_inst_pay_group
    w.stop()

    print_memory_usage(df_inst_agg, "installment_pay_aggregation_1")

    print("Processing 1")
    assert (df_inst_agg["SK_ID_CURR_min"] == df_inst_agg["SK_ID_CURR_max"]).all(axis=None), "Inconsistent SK_ID_CURR"
    df_inst_pay_processed = pd.DataFrame(index=df_inst_agg.index)
    df_inst_pay_processed["SK_ID_CURR"] = df_inst_agg["SK_ID_CURR_min"]

    df_inst_pay_group_cnt_distict = df_inst_agg["NUM_INSTALMENT_VERSION_nunique"]
    df_inst_pay_group_check = ((df_inst_pay_group_cnt == 2) |
                               (df_inst_pay_group_cnt_distict == 1))
    assert df_inst_pay_group_check.all(axis=None)
    del df_inst_pay_group_cnt, df_inst_pay_group_check
    df_inst_pay_processed["NUM_INSTALMENT_VERSION"] = df_inst_agg["NUM_INSTALMENT_VERSION_max"]

    assert (df_inst_agg["DAYS_INSTALMENT_min"] == df_inst_agg["DAYS_INSTALMENT_max"]).all(axis=None)
    df_inst_pay_processed["DAYS_INSTALMENT"] = df_inst_agg["DAYS_INSTALMENT_min"]

    df_agg_filter = (df_inst_pay_group_cnt_distict == 2)
    assert (df_agg_filter | (df_inst_agg["AMT_INSTALMENT_min"] == df_inst_agg["AMT_INSTALMENT_max"])).all(axis=None)
    df_inst_pay_processed["AMT_INSTALMENT"] = df_inst_agg["AMT_INSTALMENT_min"]
    df_inst_pay_processed.loc[df_agg_filter, "AMT_INSTALMENT"] = df_inst_agg["AMT_INSTALMENT_sum"]
    print("%d payments aggregated" % df_agg_filter.sum())
    del df_inst_pay_group_cnt_distict, df_agg_filter

    df_inst_pay_processed.reset_index(inplace=True)
    # df_inst_pay_processed["DAYS_ENTRY_PAYMENT"].astype(np.float16, copy=False)
    df_inst_pay_processed["DAYS_ENTRY_PAYMENT"] = df_inst_pay_processed["DAYS_ENTRY_PAYMENT"].astype(np.float16,
                                                                                                     copy=False)
    df_inst_pay_processed["AMT_PAYMENT"] = df_inst_pay_processed["AMT_PAYMENT"].astype(np.float32, copy=False)
    df_inst_pay_processed["AMT_PAYMENT"].replace(-1, -np.inf, inplace=True)
    assert ((df_inst_pay_processed["AMT_PAYMENT"] >= 0) |
            (df_inst_pay_processed["DAYS_ENTRY_PAYMENT"] == 0)).all(axis=None)
    df_diff_entry_offset = df_inst_pay_processed["DAYS_ENTRY_PAYMENT"] - df_inst_pay_processed["DAYS_INSTALMENT"]
    df_inst_pay_processed["AMT_DUE_PAYMENT"] = (np.fmax(df_inst_pay_processed["AMT_PAYMENT"], 0) *
                                                (df_diff_entry_offset <= 0))
    df_inst_pay_processed["AMT_DUE30_PAYMENT"] = (np.fmax(df_inst_pay_processed["AMT_PAYMENT"], 0) *
                                                  (df_diff_entry_offset <= 30))
    print_memory_usage(df_inst_pay_processed, "inst_pay_processed_1")
    # print(df_inst_pay_processed.query("(SK_ID_PREV == 1001758) & (NUM_INSTALMENT_NUMBER == 24)").transpose())

    df_inst_pay_group = df_inst_pay_processed.groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"])
    del df_diff_entry_offset, df_inst_pay_processed, df_inst_agg

    w = Watch("Aggregation 2")
    print("Aggregate multiple payments for one installment")
    w.start()
    df_inst_pay_group_cnt = df_inst_pay_group.size()
    df_inst_agg = df_inst_pay_group.agg({
        "SK_ID_CURR": ["min", "max"],
        # "NUM_INSTALMENT_VERSION": ["min", "max"],
        "DAYS_INSTALMENT": ["min", "max"],
        "DAYS_ENTRY_PAYMENT": ["min", "max"],
        "AMT_INSTALMENT": ["min", "max", "sum"],
        "AMT_PAYMENT": ["sum"],
        "AMT_DUE_PAYMENT": ["sum"],
        "AMT_DUE30_PAYMENT": ["sum"]
    }, skipna=False)
    df_inst_agg.columns = ['_'.join(col) for col in df_inst_agg.columns]
    del df_inst_pay_group
    w.stop()
    print("Finish aggregations")

    gc.collect()
    print_memory_usage(df_inst_agg, "installment_pay_aggregation_2")

    print("Processing 2")
    w = Watch("Processing 2")
    w.start()
    assert (df_inst_agg["SK_ID_CURR_min"] == df_inst_agg["SK_ID_CURR_max"]).all(), "Inconsistent SK_ID_CURR"
    df_inst_pay_processed = pd.DataFrame(index=df_inst_agg.index)
    df_inst_pay_processed["SK_ID_CURR"] = df_inst_agg["SK_ID_CURR_min"]

    # df_inst_agg_INST_VER = df_inst_agg["NUM_INSTALMENT_VERSION"]
    # assert (df_inst_agg_INST_VER["min"] == df_inst_agg_INST_VER["max"]).all(axis=None), "Inconsistent NUM_INSTALMENT_VERSION"
    # df_inst_pay_processed["NUM_INSTALMENT_VERSION"] = df_inst_agg_INST_VER["min"]

    assert (df_inst_agg["DAYS_INSTALMENT_min"] ==
            df_inst_agg["DAYS_INSTALMENT_max"]).all(axis=None), "Inconsistent DAYS_INSTALMENT"
    df_inst_pay_processed["DAYS_INSTALMENT"] = df_inst_agg["DAYS_INSTALMENT_min"]

    df_inst_pay_processed["DAYS_FIRST_PAYMENT"] = df_inst_agg["DAYS_ENTRY_PAYMENT_min"].replace(0, np.nan)
    df_inst_pay_processed["DAYS_LAST_PAYMENT"] = df_inst_agg["DAYS_ENTRY_PAYMENT_max"].replace(0, np.nan)

    assert (df_inst_agg["AMT_INSTALMENT_min"] == df_inst_agg["AMT_INSTALMENT_max"]).all(axis=None)
    df_inst_pay_processed["AMT_INSTALMENT"] = df_inst_agg["AMT_INSTALMENT_min"]

    # Fix missing installment info
    # df_prev_app_ann = pd.read_csv(r"data\previous_application.csv", index_col=0, usecols=[0, 3])
    # df_inst_agg = df_inst_agg.join(df_prev_app_ann, how="left")
    #
    # df_annuity_check = ((df_inst_agg.index.get_level_values(2) != 1) | df_inst_agg["AMT_ANNUITY"].isna() |
    #                     (df_inst_agg["AMT_INSTALMENT_min"] == 0) |
    #                     ((df_inst_agg["AMT_ANNUITY"] - df_inst_agg["AMT_INSTALMENT_min"]).abs() < 0.01))
    # assert df_annuity_check.all(axis=None)
    # inst_fix_filter = ((df_inst_agg["NUM_INSTALMENT_VERSION"] == 1) & (df_inst_agg["AMT_INSTALMENT_min"] == 0))
    # df_inst_pay_processed.loc[inst_fix_filter, "AMT_INSTALMENT"] = df_inst_agg.loc[inst_fix_filter, "AMT_ANNUITY"]
    # del df_annuity_check, inst_fix_filter

    # inst_fix_filter = (df_inst_agg["AMT_INSTALMENT_min"] == 0)
    # df_inst_pay_processed.loc[inst_fix_filter, "AMT_INSTALMENT"] = df_inst_agg.loc[inst_fix_filter, "AMT_PAYMENT_sum"]
    # del inst_fix_filter

    df_inst_pay_invalid_filter = (df_inst_agg["AMT_PAYMENT_sum"] < 0)
    assert ((~df_inst_pay_invalid_filter) | (df_inst_pay_group_cnt == 1)).all(axis=None)
    df_inst_pay_processed["AMT_PAYMENT"] = df_inst_agg["AMT_PAYMENT_sum"]
    df_inst_pay_processed.loc[df_inst_pay_invalid_filter, "AMT_PAYMENT"] = np.nan
    assert (df_inst_pay_processed["AMT_PAYMENT"] != 0).all(axis=None)

    df_inst_pay_invalid_filter = df_inst_pay_processed["AMT_PAYMENT"].isnull()
    df_inst_pay_processed["NUM_PAYMENTS"] = df_inst_pay_group_cnt.astype(np.uint16)
    df_inst_pay_processed.loc[df_inst_pay_invalid_filter, "NUM_PAYMENTS"] = np.uint16(0)
    print("%d installments aggregated" % (df_inst_pay_group_cnt > 1).sum())
    del df_inst_pay_group_cnt, df_inst_pay_invalid_filter

    df_inst_pay_processed["AMT_OVERDUE"] = np.fmax(df_inst_pay_processed["AMT_INSTALMENT"] -
                                                   df_inst_agg["AMT_DUE_PAYMENT_sum"], 0)
    df_inst_pay_processed["AMT_OVERDUE"] *= (df_inst_pay_processed["AMT_OVERDUE"] >= 0.01)
    df_inst_pay_processed["AMT_DPD30"] = np.fmax(df_inst_pay_processed["AMT_INSTALMENT"] -
                                                 df_inst_agg["AMT_DUE30_PAYMENT_sum"], 0)
    df_inst_pay_processed["AMT_DPD30"] *= (df_inst_pay_processed["AMT_DPD30"] >= 0.01)
    df_inst_pay_processed["AMT_UNPAID"] = np.fmax(df_inst_pay_processed["AMT_INSTALMENT"] -
                                                  df_inst_pay_processed["AMT_PAYMENT"].fillna(0), 0)
    df_inst_pay_processed["AMT_UNPAID"] *= (df_inst_pay_processed["AMT_UNPAID"] >= 0.01)
    df_inst_pay_processed.reset_index(inplace=True)
    # df_inst_pay_processed.rename(columns={"NUM_INSTALMENT_NUMBER": "NUM_INSTALMENT_NUMBER",
    #                                       "NUM_INSTALMENT_VERSION": "INSTALMENT_VER"})
    del df_inst_agg
    w.stop()
    print("Finish processing")

    print_memory_usage(df_inst_pay_processed, "inst_pay_processed_2")
    gc.collect()

    columns_to_write = ["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
                        "DAYS_INSTALMENT", "DAYS_FIRST_PAYMENT", "DAYS_LAST_PAYMENT", "NUM_PAYMENTS",
                        "AMT_INSTALMENT", "AMT_PAYMENT", "AMT_OVERDUE", "AMT_DPD30", "AMT_UNPAID"]

    w = Watch("Save file")
    w.start()
    df_inst_pay_processed.to_csv(r"data\installments_payments_processed.csv", index=False, columns=columns_to_write)
    w.stop()
    Watch.print_all()


def merge_payment_info():
    df_inst_pay = load_install_payments()
    df_pos = load_pos_balance()
    df_inst_pay["MONTHS_BALANCE"] = df_inst_pay["DAYS_INSTALMENT"] // 30
    df_pos.set_index(["SK_ID_PREV", "MONTHS_BALANCE"], inplace=True, drop=True)
    df_merged = df_inst_pay.join(df_pos, on=["SK_ID_PREV", "MONTHS_BALANCE"], how="outer", rsuffix="_POS")
    t = df_merged[(df_merged.NAME_CONTRACT_STATUS != "Active") & df_merged.SK_ID_CURR.notnull()]
    print(t.head())
    return t


def export_inst_pay(df_inst_pay, prev_id, path=None):
    if df_inst_pay is None:
        df_inst_pay = load_install_payments(False)
    df_inst_pay_filtered = df_inst_pay[df_inst_pay.SK_ID_PREV == prev_id].copy()
    df_inst_pay_filtered.sort_values(by=["NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION", "DAYS_ENTRY_PAYMENT"],
                                     inplace=True)
    df_inst_pay_filtered.to_csv(path if path else r"temp\inst_pay_{:d}.csv".format(prev_id), index=False)


def save_to_file(df, path):
    with open(path, "w") as f:
        df.to_string(f)


if __name__ == "__main__":
    # clean_inst_pay()
    df = merge_payment_info()
    # df_train, df_test = load_app_data()
    # print_memory_usage(df_train, "app train")
    # print_memory_usage(df_test, "app test")
    # df = load_prev_app_data()
    # print_memory_usage(df, "prev app")
    # df = load_credit_balance()
    # print_memory_usage(df, "credit balance")
    # df_pos = load_pos_balance()
    # print_memory_usage(df_pos, "pos balance")
