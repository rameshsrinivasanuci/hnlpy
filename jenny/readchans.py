def getchans():
    with open('/home/jenny/hnlpy/jenny/channel_ind.txt') as f:
        lines = f.readlines()
    Chans ={}
    for i in lines:
        chan  = i[0:-2]
        chan = chan.split(" = ")
        vals =chan[1][1:-1]
        vals= list(map(int, vals.split()))
        vals = [i-1 for i in vals]
        name = chan[0].strip()
        Chans[name]= vals
    return Chans



