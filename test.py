base = 3600000
byt= 9600000
byt_val = 0
hypo = 32000
interest = 0.08
byt_interest = 0.06
month_interest = interest / 12
years = 30
val = 0
rent = 25000
rent_interest = 0.025

all_years = []

def get_diff_interest(diff):
    #calculate interest per each month
    year_total = 0
    for i in range(0,12):
        remaining_months = 12 - i
        #get the interest for the remaining months
        appr_percent = (1+month_interest)**remaining_months
        month_appr = diff * appr_percent
        print('month_appr',month_appr)
        year_total = year_total + month_appr
    print('year_total',year_total)
    return year_total

get_diff_interest(5000)


paid_hypo = 0

for i in range(0,years):
    print(i)
    paid_rent_year = (rent * 12) * (1+rent_interest)

    rent = paid_rent_year / 12

    #print('paid rent',paid_rent_year)
    hypo_year = hypo * 12
    #print('hypo year',hypo_year)

    diff = hypo_year - paid_rent_year
    #print('diff',diff)
    if byt_val == 0:
        byt_val = (byt * (1+byt_interest))
    else:
        byt_val = (byt_val * (1+byt_interest))


    if diff < 0:
        hypo_year = 0
        diff = 0
        print('RENT is bigger than hypo!')
        #exit(1)
        year_appr = 0
    else:
        year_appr = get_diff_interest(diff/12)


    paid_hypo = paid_hypo + hypo_year
    if val == 0:
        val = (base * (1+interest)) + year_appr
    else:
        val = (val * (1+interest)) + year_appr
        #print('increased val',val)
    #print('value',val)
    #print('paid_hypo',paid_hypo)

    metrics = {
        'year': i,
        'paid_hypo': paid_hypo,
        'rent': rent,
        'byt_val': byt_val,
        'value': val,
        'diff': diff,
        'year_diff_appr': year_appr
    }
    print('metrics',metrics)
    all_years.append(metrics)
    #exit(1)
import pandas as pd
df = pd.DataFrame(all_years)
df.to_excel('hypo.xlsx')
print('final value',val)