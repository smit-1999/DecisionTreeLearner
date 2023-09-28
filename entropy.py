def calculate_entropy(df):
    # once we have the possible splits we will try to find entropy for the same
    # h_y = -p0*logp0 - p1*logp1
    # where p0=countOfZeros/totalValues, p1=countOfOnes/totalValues
    c0 = 0
    c1 = 0
    if df['Y'].value_counts().get(0, 0):
        c0 = df['Y'].value_counts()[0]
    if df['Y'].value_counts().get(1, 0):
        c1 = df['Y'].value_counts()[1]
    if c0 == 0:
        return 0
    if c1 == 0:
        return 0
    c_total = c0 + c1

    p0 = c0/c_total
    log_2_p0 = math.log2(p0)

    p1 = c1/c_total
    log_2_p1 = math.log2(p1)

    h_y = (-p0*log_2_p0) - (p1*log_2_p1)
    return h_y
