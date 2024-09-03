base = 3600000
hypo = 31000
interest = 0.08
years = 30
val = 0
rent = 26000
rent_interest = 0.03

paid_hypo = 0

for i in range(0,years):
    print(i)
    paid_rent_year = (rent * 12) * (1+rent_interest)

    rent = paid_rent_year / 12

    print('paid rent',paid_rent_year)
    hypo_year = hypo * 12
    print('hypo year',hypo_year)

    diff = hypo_year - paid_rent_year
    print('diff',diff)


    if diff < 0:
        hypo_year = 0
        print('RENT is bigger than hypo!')
        #exit(1)


    paid_hypo = paid_hypo + hypo_year
    if val == 0:
        val = (base * (1+interest)) + (diff * (1+interest))
    else:
        val = val * (1+interest)
        print('increased val',val)
        val = val + (diff * (1+interest)) if diff > 0 else val
    print('value',val)
    print('paid_hypo',paid_hypo)
    exit(1)


