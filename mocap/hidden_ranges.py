import html
SMOOTH_MAX_FRAMES=30
AUTO_MAX_FRAMES=290  # with 5 known frames on each side the generator's 300-frame limit
METHODS=('none','smooth','copy','auto')
SEED_LIMIT=999
OFFSET_LIMIT=600
TEXT_LIMIT=200  # characters of an automatic fill's description

def printable(text):
    """The text without control and other unprintable characters; whitespace is kept for the caller to collapse."""
    return ''.join(c for c in text if c.isprintable() or c.isspace())

def normalise_text(text):
    """A description as one line: unprintable characters dropped, every run of whitespace, line breaks included, as one
    space; ValueError above TEXT_LIMIT characters."""
    text=' '.join(printable(text).split())
    if len(text)>TEXT_LIMIT:raise ValueError(f'A description can have at most {TEXT_LIMIT} characters.')
    return text

def typed(text):
    """`text` for a message, so that the Console and the status line show it as typed. Both read a line with a tag or
    `&lt;` in it as rich text and any other line as it is, so only such text is escaped, inside a tag that settles it."""
    return '<span>'+html.escape(text,quote=False)+'</span>' if '<' in text or '&lt;' in text else text

def gaps(takes):
    out=[];prev=None
    for t in sorted(takes,key=lambda t:(t['first'],t['end'])):
        if prev is not None and t['first']>prev['end']:out.append(dict(before=prev,after=t,first=prev['end'],end=t['first']))
        if prev is None or t['end']>prev['end']:prev=t
    return out

def gap_key(gap):return f"{gap['first']}_{gap['end']}"

def choice_params(choice):
    """(method, params) of a gap choice record, normalised; ValueError for a malformed one."""
    if choice is None or choice=={}:return 'none',{}
    if not isinstance(choice,dict):raise ValueError('Invalid gap choice')
    m=choice.get('method','none')
    if m not in METHODS:raise ValueError(f'Unknown method: {m}')
    if m=='auto':
        seed=choice.get('seed',0)
        text=choice.get('text','')
        if type(seed) is not int or not 0<=seed<=SEED_LIMIT or not isinstance(text,str):raise ValueError('Invalid automatic choice')
        text=normalise_text(text)  # without a description the params are those of step 4, so its joins stay current
        return m,dict(seed=seed,text=text) if text else dict(seed=seed)
    if m!='copy':return m,{}
    p,mirror,off=choice.get('person'),choice.get('mirror',False),choice.get('offset',0)
    if type(p) is not int or type(mirror) is not bool or type(off) is not int or abs(off)>OFFSET_LIMIT:raise ValueError('Invalid copy choice')
    return m,dict(person=p,mirror=mirror,offset=off)

def donor_take(takes,gap,offset):
    """Last source take containing the frames just before and after the gap, shifted by offset."""
    lo,hi=gap['first']-1+offset,gap['end']+1+offset;found=None
    for t in takes:
        if isinstance(t,dict) and type(t.get('first')) is int and type(t.get('end')) is int and t['first']<=lo and t['end']>=hi:found=t
    return found

def donor_offsets(takes,gap):
    """Offsets, as merged (lowest, highest) spans, at which one take covers the gap and a frame on each side."""
    spans=[]
    for t in takes if isinstance(takes,list) else []:
        if not (isinstance(t,dict) and type(t.get('first')) is int and type(t.get('end')) is int):continue
        lo,hi=max(-OFFSET_LIMIT,t['first']-gap['first']+1),min(OFFSET_LIMIT,t['end']-gap['end']-1)
        if lo<=hi:spans.append((lo,hi))
    out=[]
    for lo,hi in sorted(spans):
        if out and lo<=out[-1][1]+1:out[-1]=(out[-1][0],max(hi,out[-1][1]))
        else:out.append((lo,hi))
    return out

def offset_hint(spans,most=3):
    """The spans nearest to no shift, as the sentence that follows a refused offset."""
    near=sorted(sorted(spans,key=lambda s:0 if s[0]<=0<=s[1] else min(abs(s[0]),abs(s[1])))[:most])
    return ' Offsets that work: '+', '.join(str(lo) if lo==hi else f'{lo} to {hi}' for lo,hi in near)+'.'

def method_issue(method,gap,choice=None,people=None,ident=None):
    if method=='none':return None
    if method=='copy':
        ask='Choose a person to copy from.'
        try:p=choice_params(dict(choice or {},method='copy'))[1]
        except (ValueError,TypeError):return ask
        donor=next((x for x in people or [] if isinstance(x,dict) and x.get('id')==p['person']),None)
        if donor is None or p['person']==ident:return ask
        takes=donor.get('takes');name=donor.get('name') or f"Person {p['person']}"
        if donor_take(takes if isinstance(takes,list) else [],gap,p['offset']) is None:
            spans=donor_offsets(takes,gap)
            if not spans:return f'{name} has no captured take long enough to cover this range.'
            return f'{name} has no captured motion covering this range with this offset.'+offset_hint(spans)
        return None
    n=gap['end']-gap['first']
    if method=='auto':return f'Automatic fill is limited to {AUTO_MAX_FRAMES} frames; this range has {n}.' if n>AUTO_MAX_FRAMES else None
    if method!='smooth':raise ValueError(f'Unknown method: {method}')
    return f'Smooth join is limited to {SMOOTH_MAX_FRAMES} frames; this range has {n}.' if n>SMOOTH_MAX_FRAMES else None

def chains(takes,choices,people=None,ident=None):
    out=[];cur=[]
    for g in gaps(takes):
        try:
            c=choices.get(gap_key(g));m=choice_params(c)[0]
            ok=m!='none' and method_issue(m,g,c,people,ident) is None
        except Exception:ok=False
        if ok:
            if not cur or cur[-1] is not g['before']:
                if cur:out.append(cur)
                cur=[g['before']]
            cur.append(g['after'])
        elif cur:out.append(cur);cur=[]
    if cur:out.append(cur)
    return out

def gap_label(gap,fps):return f"Frames {gap['first']+1}–{gap['end']} ({(gap['end']-gap['first'])/fps:.1f} s)"
