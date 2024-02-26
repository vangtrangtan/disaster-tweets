
# ['x', 'a', 'x', 'y', 'a', 'x'] -> [('x', 3), ('a', 2), ('y', 1)]
def group_list(lst):
    freq_dict = {}
    for el in lst:
        if el in freq_dict:
            freq_dict[el] += 1
        else:
            freq_dict[el] = 1
    res = list(freq_dict.items())
    res.sort(key=lambda x: x[1], reverse=True)
    return res
