SMOOTH_MAX_FRAMES=30
METHODS=('none','smooth')

def gaps(takes):
    out=[];prev=None
    for t in sorted(takes,key=lambda t:(t['first'],t['end'])):
        if prev is not None and t['first']>prev['end']:out.append(dict(before=prev,after=t,first=prev['end'],end=t['first']))
        if prev is None or t['end']>prev['end']:prev=t
    return out

def gap_key(gap):return f"{gap['first']}_{gap['end']}"

def method_issue(method,gap):
    if method=='none':return None
    if method!='smooth':raise ValueError(f'Unknown method: {method}')
    n=gap['end']-gap['first']
    return f'Smooth join is limited to {SMOOTH_MAX_FRAMES} frames; this range has {n}.' if n>SMOOTH_MAX_FRAMES else None

def chains(takes,choices):
    out=[];cur=[]
    for g in gaps(takes):
        m=(choices.get(gap_key(g)) or {}).get('method','none')
        if m!='none' and method_issue(m,g) is None:
            if not cur or cur[-1] is not g['before']:
                if cur:out.append(cur)
                cur=[g['before']]
            cur.append(g['after'])
        elif cur:out.append(cur);cur=[]
    if cur:out.append(cur)
    return out

def gap_label(gap,fps):return f"Frames {gap['first']+1}–{gap['end']} ({(gap['end']-gap['first'])/fps:.1f} s)"
