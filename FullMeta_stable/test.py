import csv

import math

def FNPS(T):
    # Note: Missing '(' in document
    return math.exp(16.6536 - 4030.183 / (T + 235.0))

def computePMV(TA, VEL, RH, MET, CLO, PA):
    if PA == 0:
        PA = RH * 10 * FNPS(TA)  # water vapour pressure, Pa
    ICL = 0.155 * CLO  # thermal insulation of the clothing in m2K/W
    M = MET * 58.15  # external work in W/m2
    TR = TA
    #   W = WME * 58.15
    MW = M  # internal heat production in the human body
    if ICL <= 0.078:
        FCL = 1 + 1.29 * ICL
    else:
        FCL = 1.05 + 0.645 * ICL  # clothing area factor
    HCF = 12.1 * math.sqrt(VEL)  # heat transf. coeff. by forced convection
    TAA = TA + 273  # air temperature in Kelvin
    TRA = TR + 273  # mean radiant temperature in Kelvin

    TCLA = TAA + (35.5 - TA) / (3.5 * ICL + 0.1)  # first guess for surface temperature of clothing
    P1 = ICL * FCL
    P2 = P1 * 3.96
    P3 = P1 * 100
    P4 = P1 * TAA
    # Note: P5 = 308.7 - 0.028 * MW + P2 * (TRA / 100) * 4  in document
    P5 = (308.7 - 0.028 * MW) + (P2 * math.pow(TRA / 100, 4))
    # Note: TLCA in document
    XN = TCLA / 100
    # Note: XF = XN in document
    XF = TCLA / 50
    N = 0  # number of iterations
    EPS = 0.00015  # stop criteria in iteration
    # Note: HC must be defined before use
    HC = HCF

    while abs(XN - XF) > EPS:
        XF = (XF + XN) / 2
        HCN = 2.38 * math.pow(abs(100.0 * XF - TAA), 0.25)
        if HCF > HCN:
            HC = HCF
        else:
            HC = HCN
        # Note: should be '-' in document
        XN = (P5 + P4 * HC - P2 * math.pow(XF, 4)) / (100 + P3 * HC)
        N = N + 1
        if N > 150:
            print('Max iterations exceeded')
            return 999999
    TCL = 100 * XN - 273

    HL1 = 3.05 * 0.001 * (5733 - 6.99 * MW - PA)  # heat loss diff. through skin
    if MW > 58.15:
        HL2 = 0.42 * (MW - 58.15)
    else:
        HL2 = 0
    HL3 = 1.7 * 0.00001 * M * (5867 - PA)  # latent respiration heat loss
    HL4 = 0.0014 * M * (34 - TA)  # dry respiration heat loss
    # Note: HL5 = 3.96 * FCL * (XN^4 - (TRA/100^4)   in document
    HL5 = 3.96 * FCL * (math.pow(XN, 4) - math.pow(TRA / 100, 4))  # heat loss by radiation
    HL6 = FCL * HC * (TCL - TA)

    TS = 0.303 * math.exp(-0.036 * M) + 0.028
    PMV = TS * (MW - HL1 - HL2 - HL3 - HL4 - HL5 - HL6)
    # PMV = abs(PMV)
    # PMV=10-PMV
    return PMV  # PMV越小越好

if __name__ == '__main__':
    CLO_L = [
        0,
        1.34,
        1.18,
        0.83,
        0.59,
        0.41,
        0.33,
        0.31,
        0.31,
        0.44,
        0.51,
        0.76,
        1.26
    ]

    # pmv_data=[]
    # for month in range(12):
    #     monthly_pmv_data = []
    #     TA = -4
    #     for _ in range(44000):
    #         TA+=0.001
    #         DEFAULT_VEL=0.1
    #         DEFAULT_RH=50
    #         DEFAULT_MET=1
    #         DEFAULT_PA=0
    #         people_effect=0
    #         pmv = computePMV(TA, DEFAULT_VEL, DEFAULT_RH, DEFAULT_MET + people_effect, CLO_L[month], DEFAULT_PA)
    #         monthly_pmv_data.append(pmv)
    #     pmv_data.append(monthly_pmv_data)
    #
    # # 将pmv_data保存为CSV文件
    # csv_file_name = 'pmv_data.csv'
    # with open(csv_file_name, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     for monthly_data in pmv_data:
    #         csvwriter.writerow(monthly_data)
    #
    # print(f'数据已保存到 {csv_file_name}')

    for _ in range(520000*8*4):
        pmv = computePMV(-4, 0.1, 50, 1, 0.59, 0)