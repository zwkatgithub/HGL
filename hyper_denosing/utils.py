def lines_generator(fin, num=4):
    lines = []
    for line in fin:
        lines.append(line.strip())
        if len(lines)==num:
            yield lines
            lines = []
    if len(lines)==num:
        yield lines

def spans_generator(line, sep=' | '):
    if line.strip()=="":
        return list()
    spans = line.strip().split(sep)
    for span in spans:
        se, type_ = span.split()
        start, end = list(map(int, se.split(",")))
        yield (start, end, type_)
    

def convert_hat(words, start, end, start_tag='<E>', end_tag='</E>'):
    ret = words[:]
    ret.insert(start, start_tag)
    ret.insert(end+2, end_tag)
    return ret, ret.index(start_tag), ret.index(end_tag)