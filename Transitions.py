def transition(beta, sig, conf, transi):
    if transi=='LA':
        left_arc(beta, sig, conf)
    if transi=='RA':
        right_arc(beta, sig, conf)
    if transi =='RE':
        reduce(conf)
    if transi=='SH':
        shift(conf)

def left_arc(beta, sig, conf):
    wi = conf.stack.pop()
    dep = (beta, sig.getFeat('LABEL'), wi)
    conf.dependencies.append(dep)

def right_arc(beta, sig, conf):
    wj = conf.buffer.pop(0)
    conf.stack.append(wj)
    dep = (sig, beta.getFeat('LABEL'), wj)
    conf.dependencies.append(dep)

def reduce(conf):
    conf.stack.pop()

def shift(conf):
    wi = conf.buffer.pop(0)
    conf.stack.append(wi)
