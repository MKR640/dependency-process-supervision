"""Reconstruct the paper's method figure as editable SVG, vector PDF and PNG.

Install optional rendering dependencies: pip install reportlab pypdfium2
Run from the repository root: python scripts/render_method_overview.py
No source bitmap is embedded or traced. Icons, text, and connectors are primitives.
"""
import argparse
from html import escape
import math
import os
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

W, H = 1672, 941
BLUE, NAVY, GREY, RED = '#0000ed', '#263d99', '#ababab', '#ff2020'
GRADIENTS = {
    'blue': ('#f1faff', '#cee9ff'),
    'red': ('#fffafa', '#ffecec'),
    'orange': ('#fffaf1', '#ffedd7'),
    'green': ('#f8fff6', '#e8f7e3'),
}


def fonts():
    windows = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    linux = Path('/usr/share/fonts/truetype/liberation2')
    for key, win, lin, fallback in [
        ('r', 'times.ttf', 'LiberationSerif-Regular.ttf', 'Times-Roman'),
        ('b', 'timesbd.ttf', 'LiberationSerif-Bold.ttf', 'Times-Bold'),
        ('i', 'timesi.ttf', 'LiberationSerif-Italic.ttf', 'Times-Italic'),
        ('bi', 'timesbi.ttf', 'LiberationSerif-BoldItalic.ttf', 'Times-BoldItalic'),
    ]:
        path = next((p for p in [windows / win, linux / lin] if p.is_file()), None)
        if path:
            name = 'FigureSerif-' + key
            pdfmetrics.registerFont(TTFont(name, str(path)))
            FONT[key] = name
        else:
            FONT[key] = fallback
    symbol=next((p for p in [windows/'seguisym.ttf',Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.is_file()),None)
    if symbol:
        pdfmetrics.registerFont(TTFont('FigureSymbols',str(symbol)))
        FONT['symbol']='FigureSymbols'


def text_runs(text,style):
    result=[]
    primary=FONT[style]
    for char in text:
        face=pdfmetrics.getFont(primary).face
        font=primary
        if hasattr(face,'charToGlyph') and ord(char) not in face.charToGlyph:
            font=FONT.get('symbol',primary)
            fallback=pdfmetrics.getFont(font).face
            if hasattr(fallback,'charToGlyph') and ord(char) not in fallback.charToGlyph:
                raise ValueError(f'No installed font supports character U+{ord(char):04X}')
        if result and result[-1][1]==font:
            result[-1]=(result[-1][0]+char,font)
        else:
            result.append((char,font))
    return result


FONT = {}


class Figure:
    def __init__(self, pdf_path):
        self.pdf = canvas.Canvas(str(pdf_path), pagesize=(W, H), pageCompression=1)
        self.pdf.setTitle('Dependency-grounded process supervision: method overview')
        self.pdf.setAuthor('Anonymous')
        self.pdf.setCreator('Vector figure renderer')
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
                      '<title id="title">Dependency-grounded process supervision</title>',
                      '<desc id="desc">Vector reconstruction of the method overview. Editable text, paths and icons; no embedded bitmap.</desc>', '<defs>']
        for name, (a, b) in GRADIENTS.items():
            self.parts.append(f'<linearGradient id="{name}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>')
        self.parts.append('</defs>')
        self.rect(0, 0, W, H, fill='#ffffff', stroke=None, radius=0)

    def group(self, name):
        self.parts.append(f'<g id="{name}">')

    def end(self):
        self.parts.append('</g>')

    def rect(self, x, y, w, h, fill='none', stroke='#a6a6a6', radius=7, width=1, dash=False):
        sf = f'url(#{fill})' if fill in GRADIENTS else fill
        d = ' stroke-dasharray="4 3"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{sf}" stroke="{stroke or "none"}" stroke-width="{width}"{d}/>')
        c = self.pdf
        c.saveState()
        if fill in GRADIENTS:
            p = c.beginPath(); p.roundRect(x, H-y-h, w, h, radius)
            c.clipPath(p, stroke=0, fill=0)
            c.linearGradient(x, H-y, x+w, H-y-h, [HexColor(t) for t in GRADIENTS[fill]], extend=True)
        elif fill != 'none':
            c.setFillColor(HexColor(fill)); c.roundRect(x, H-y-h, w, h, radius, stroke=0, fill=1)
        c.restoreState()
        if stroke:
            c.saveState(); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(width)
            if dash: c.setDash(4, 3)
            c.roundRect(x, H-y-h, w, h, radius, stroke=1, fill=0); c.restoreState()

    def path(self, commands, stroke=GREY, width=2, fill='none', dash=False):
        data = ' '.join(op + ' ' + ' '.join(str(v) for v in values) for op, values in commands)
        d = ' stroke-dasharray="6 3"' if dash else ''
        self.parts.append(f'<path d="{data}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{d}/>')
        c = self.pdf; c.saveState(); p = c.beginPath()
        for op, v in commands:
            if op == 'M': p.moveTo(v[0], H-v[1])
            elif op == 'L': p.lineTo(v[0], H-v[1])
            elif op == 'C': p.curveTo(v[0], H-v[1], v[2], H-v[3], v[4], H-v[5])
            elif op == 'Z': p.close()
        c.setLineWidth(width); c.setLineJoin(1); c.setLineCap(1)
        if dash: c.setDash(6, 3)
        if stroke: c.setStrokeColor(HexColor(stroke))
        if fill != 'none': c.setFillColor(HexColor(fill))
        c.drawPath(p, stroke=int(bool(stroke)), fill=int(fill != 'none')); c.restoreState()

    def circle(self, x, y, r, fill=NAVY, stroke=None, width=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
        c=self.pdf; c.saveState(); c.setFillColor(HexColor(fill)); c.setLineWidth(width)
        if stroke: c.setStrokeColor(HexColor(stroke))
        c.circle(x, H-y, r, stroke=int(bool(stroke)), fill=1); c.restoreState()

    def text(self, x, y, text, size=23, style='r', color='#000000', anchor='start', max_width=None):
        # Fit by font metrics, not raster resampling or distorted text scaling.
        runs=text_runs(text,style)
        measured=sum(pdfmetrics.stringWidth(t,font,size) for t,font in runs)
        if max_width and measured > max_width: size *= max_width/measured
        measured=sum(pdfmetrics.stringWidth(t,font,size) for t,font in runs)
        left=x - (measured/2 if anchor=='middle' else measured if anchor=='end' else 0)
        weight='bold' if 'b' in style else 'normal'
        slant='italic' if 'i' in style else 'normal'
        spans=''.join(escape(t) if font==FONT[style] else f'<tspan font-family="Segoe UI Symbol, DejaVu Sans, sans-serif" font-style="normal" font-weight="normal">{escape(t)}</tspan>' for t,font in runs)
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Times New Roman, Liberation Serif, serif" font-size="{size:.3f}" font-weight="{weight}" font-style="{slant}" fill="{color}" text-anchor="{anchor}">{spans}</text>')
        c=self.pdf; c.setFillColor(HexColor(color))
        for t,font in runs:
            c.setFont(font,size); c.drawString(left,H-y,t)
            left+=pdfmetrics.stringWidth(t,font,size)

    def subscript(self,x,y,base,sub,size=23,style='bi'):
        small=size*.68
        a=pdfmetrics.stringWidth(base,FONT[style],size)
        b=pdfmetrics.stringWidth(sub,FONT[style],small)
        left=x-(a+b)/2
        self.text(left,y,base,size,style)
        self.text(left+a,y+size*.2,sub,small,style)

    def arrow(self, points, color=GREY, width=4, dash=False, head=12):
        self.path([('M',points[0])]+[('L',p) for p in points[1:]],stroke=color,width=width,dash=dash)
        x,y=points[-1]; px,py=points[-2]; a=math.atan2(y-py,x-px)
        bx,by=x-head*math.cos(a),y-head*math.sin(a)
        dx,dy=head*.50*math.sin(a),-head*.50*math.cos(a)
        self.path([('M',(x,y)),('L',(bx+dx,by+dy)),('L',(bx-dx,by-dy)),('Z',())],stroke=None,fill=color)

    def objective(self,x,y):
        pieces=[('ℒ = ℒ',34,0),('C',22,7),(' + 0.5ℒ',34,0),('R',22,7),(' + 0.5ℒ',34,0),('T',22,7)]
        widths=[pdfmetrics.stringWidth(t,FONT['i'],s) for t,s,_ in pieces]
        left=x-sum(widths)/2
        self.group('training-objective')
        for (t,s,dy),width in zip(pieces,widths):
            self.text(left,y+dy,t,s,'i'); left+=width
        self.end()

    def robot(self, x, y, scale=1, name='robot'):
        self.group(name)
        def r(a,b,w,h,**kw): self.rect(x+a*scale,y+b*scale,w*scale,h*scale,**kw)
        self.path([('M',(x+38*scale,y+8*scale)),('L',(x+38*scale,y+19*scale))],stroke=NAVY,width=4*scale)
        self.circle(x+38*scale,y+7*scale,5*scale,fill=NAVY)
        r(1,29,10,20,fill='#367ac5',stroke=NAVY,radius=3*scale,width=2*scale)
        r(65,29,10,20,fill='#367ac5',stroke=NAVY,radius=3*scale,width=2*scale)
        r(11,20,55,44,fill='#c5def6',stroke=None,radius=8*scale)
        r(9,18,56,43,fill='#ffffff',stroke=NAVY,radius=8*scale,width=2.7*scale)
        self.circle(x+25*scale,y+36*scale,3.6*scale)
        self.circle(x+49*scale,y+36*scale,3.6*scale)
        self.path([('M',(x+29*scale,y+47*scale)),('C',(x+33*scale,y+51*scale,x+41*scale,y+51*scale,x+45*scale,y+47*scale))],stroke=NAVY,width=2.5*scale)
        self.end()

    def document(self, x, y, w, h, name='document', lines=True):
        self.group(name); fold=w*.25
        self.path([('M',(x+5,y)),('L',(x+w-fold,y)),('L',(x+w,y+fold)),('L',(x+w,y+h-5)),('C',(x+w,y+h,x+w-2,y+h,x+w-5,y+h)),('L',(x+5,y+h)),('C',(x,y+h,x,y+h-2,x,y+h-5)),('L',(x,y+5)),('C',(x,y,x+2,y,x+5,y)),('Z',())],stroke=NAVY,width=2,fill='#ffffff')
        self.path([('M',(x+w-fold,y)),('L',(x+w-fold,y+fold)),('L',(x+w,y+fold))],stroke=NAVY,width=1.5,fill='#d6ecff')
        if lines:
            for iy, end in [(h*.46,w*.73),(h*.65,w*.73),(h*.83,w*.57)]:
                self.path([('M',(x+w*.21,y+iy)),('L',(x+end,y+iy))],stroke=NAVY,width=2)
        self.end()

    def stack(self,x,y,w=80,h=90,labels=('s₁','s₂','s₃','⋯'),name='reasoning-pages'):
        self.group(name)
        for dx,dy in [(10,0),(5,5),(0,10)]:
            self.rect(x+dx,y+dy,w-10,h-10,fill='#ffffff',stroke=NAVY,radius=8,width=1.8)
        for i,label in enumerate(labels):
            if label.startswith('s'):
                number=label[1:].translate(str.maketrans('₁₂₃ₙ','123N'))
                self.subscript(x+(w-10)/2,y+29+i*17,'s',number,22)
            else:
                self.text(x+(w-10)/2,y+29+i*17,label,size=19,style='bi',anchor='middle')
        self.end()

    def save(self,svg_path):
        self.parts.append('</svg>')
        svg_path.write_text('\n'.join(self.parts)+'\n',encoding='utf-8')
        self.pdf.showPage(); self.pdf.save()


def build(f):
    # Coordinates follow the supplied 1672 x 941 layout.
    f.group('annotation-panel')
    f.rect(13,11,1648,571,stroke='#777777',width=1)
    f.text(22,46,'(a) Evidence-Preserving Annotation',34,'b')
    for x,w in [(21,482),(510,696),(1213,440)]: f.rect(x,60,w,515,dash=True)
    f.text(262,89,'Preserve Evidence',29,'bi',BLUE,'middle')
    f.text(858,88,'Attribute Errors via Consumed Content',29,'bi',BLUE,'middle',670)
    f.text(1433,89,'Construct Supervision',29,'bi',BLUE,'middle')

    f.group('preserve-evidence')
    for yy,label,n in [(126,'Problem x','problem'),(236,'Solution R','solution')]:
        f.rect(36,yy,172,96,'blue','#59aeff')
        f.document(51,yy+22,43,56,name=n+'-icon')
        f.text(106,yy+65,label,25,'i',BLUE,max_width=95)
    f.path([('M',(210,166)),('L',(220,166)),('C',(228,166,228,174,228,183)),('L',(228,207)),('C',(228,215,233,216,241,216))],width=4)
    f.path([('M',(210,281)),('L',(220,281)),('C',(228,281,228,274,228,264)),('L',(228,229)),('C',(228,221,233,216,241,216))],width=4)
    f.arrow([(241,216),(278,216)],width=5,head=15)
    f.rect(283,165,203,89,'blue','#59aeff')
    f.text(384.5,204,'SaT spans',27,'b',anchor='middle')
    f.text(384.5,234,'→ fixed steps',26,'b',anchor='middle')
    for i,fill in enumerate(['#fff0db','#e3f4ff','#e6f5e1','#fff1e8']):
        f.rect(32+i*52,353,52,46,fill,['#ffc27c','#68baff','#9aca8d','#ffaaaa'][i],radius=3)
        f.text(58+i*52,382,'…',23,anchor='middle')
    f.arrow([(245,375),(273,375)],width=5,head=14)
    for xx,ww,t in [(278,47,'s₁'),(331,47,'s₂'),(384,47,'s₃'),(437,56,'⋯')]:
        f.rect(xx,350,ww,49,'blue','#5bb0ff',radius=5)
        f.text(xx+ww/2,382,t,28,'bi',anchor='middle')
    f.text(132,427,'Contiguous text spans',23,anchor='middle')
    f.text(385,427,'Numbered semantic steps',23,anchor='middle',max_width=220)
    f.rect(32,491,462,70,'blue',None)
    f.text(263,533,'Preserve text · Fix boundaries',29,'bi',BLUE,'middle',432)
    f.end()

    f.group('consumed-content-attribution')
    f.rect(527,96,665,92,'blue','#59aeff')
    f.text(544,123,'Thin-film interference · real trace excerpt',25,'bi',BLUE)
    f.text(543,151,'Normal incidence: air / oil (n = 1.5, t = 500 nm) / water (n = 1.33)',22,max_width=635)
    f.text(543,177,'Which wavelength is most strongly reflected?',23,max_width=423)
    f.text(1181,181,'Selected steps; text condensed',18,'i',anchor='end',max_width=212)
    f.rect(528,197,664,90,'red','#ff9898',radius=6)
    f.text(548,232,'s₂:',25,'bi')
    f.rect(600,210,269,40,'blue','#59aeff',radius=5)
    f.text(734.5,238,'Optical length = n × distance',22,anchor='middle',max_width=255)
    f.rect(888,207,294,57,'red','#ff7777',radius=5)
    f.text(1035,231,'Constructive interference: Δ = mλ',21,anchor='middle',max_width=280)
    f.rect(964,238,142,24,RED,None,radius=5)
    f.text(1035,257,'New error',21,color='#ffffff',anchor='middle')
    f.text(1035,283,'omits reflection phase shift',21,color=RED,anchor='middle')
    f.arrow([(710,251),(710,312)],head=15)
    f.arrow([(1027,287),(1027,312)],RED,3,True,14)
    f.rect(584,315,240,65,'blue','#59aeff',radius=6)
    f.text(704,342,'s₃:  L = nt = 750 nm',24,'i',anchor='middle',max_width=222)
    f.rect(642,350,125,25,'#219fff',None,radius=5)
    f.text(704,370,'Correct',22,color='#ffffff',anchor='middle')
    f.rect(911,315,229,65,'orange','#ffb455',radius=6)
    f.text(1025,342,'s₄:  2L = mλ',24,'i',anchor='middle')
    f.rect(963,350,133,25,'#ff900c',None,radius=5)
    f.text(1029,370,'Propagated',21,color='#ffffff',anchor='middle')
    f.arrow([(686,381),(686,419)],head=15)
    f.arrow([(909,364),(808,422)],RED,3,True,12)
    f.arrow([(1034,381),(1034,410)],RED,3,True,14)
    f.rect(530,422,279,69,'orange','#ffb455',radius=6)
    f.text(669,451,'s₆:  λ = 500 nm ⇒ m = 3',24,'i',anchor='middle',max_width=262)
    f.rect(613,459,113,26,'#ff900c',None,radius=5)
    f.text(669.5,480,'Propagated',21,color='#ffffff',anchor='middle')
    f.rect(859,411,333,86,'red','#ff9191',radius=6)
    f.text(1025.5,440,'s₅:  Smallest m ⇒ strongest reflection',22,anchor='middle',max_width=312)
    f.rect(941,447,175,25,RED,None,radius=5)
    f.text(1028.5,467,'New error (mixed)',21,color='#ffffff',anchor='middle')
    f.text(1026,489,'inherits wrong condition + adds invalid criterion',18,'i',RED,'middle',316)
    f.rect(548,508,616,31,stroke='#bababa',radius=4,dash=True)
    f.arrow([(583,524),(641,524)],width=3,head=12)
    f.text(649,532,'Consumed content',21)
    f.arrow([(844,524),(903,524)],RED,3,True,12)
    f.text(911,532,'Error-carrying dependency',21,max_width=244)
    f.rect(520,541,673,31,'blue',None,radius=5)
    f.text(856.5,565,'An erroneous step can contain a valid output.',24,'bi',BLUE,'middle',648)
    f.end()

    f.group('supervision-construction')
    for yy,text in [(114,'Structural consistency'),(198,'Fully evaluable traces')]:
        f.rect(1250,yy,263,47,'blue','#59aeff',radius=6)
        f.text(1381.5,yy+32,text,25,color=BLUE,anchor='middle',max_width=247)
    f.arrow([(1382,162),(1382,195)],width=5,head=16)
    f.arrow([(1382,246),(1382,282)],width=5,head=16)
    f.rect(1224,285,417,149,'red','#ff9fa4',radius=6)
    f.text(1243,316,'Target for each step:',27,'bi',BLUE)
    for yy,left,right in [(350,'Correctness:','correct / incorrect'),(385,'Role:','correct / new / propagated'),(417,'Type:','known new errors only')]:
        f.text(1265,yy,left,25,'b')
        f.text(1416,yy,right,23,max_width=210)
    f.rect(1224,462,417,76,'blue',None)
    f.text(1432.5,491,'Recovery → correct role',27,'bi',BLUE,'middle',390)
    f.text(1432.5,523,'Unknown type → type loss masked',24,'bi',BLUE,'middle',395)
    f.end(); f.end()

    f.group('hierarchical-prm-training')
    f.rect(13,591,1103,335,stroke='#777777')
    f.text(22,627,'(b) Hierarchical PRM Training',34,'b')
    # The large input page and its small document emblem mirror the reference.
    f.path([('M',(45,659)),('L',(151,659)),('L',(180,689)),('L',(180,795)),('C',(180,802,177,802,172,802)),('L',(46,802)),('C',(39,802,39,799,39,794)),('L',(39,665)),('C',(39,659,41,659,45,659)),('Z',())],stroke='#59aeff',width=1,fill='#e6f4ff')
    f.path([('M',(151,659)),('L',(151,689)),('L',(180,689))],stroke='#59aeff',width=1,fill='#d0eaff')
    f.document(56,674,42,43,name='target-step-document')
    for yy,t in [(743,'Problem +'),(767,'prefix +'),(791,'target step')]: f.text(106,yy,t,26,anchor='middle')
    f.text(120,829,'Text-only student input',22,anchor='middle')
    f.arrow([(184,727),(228,727)],width=5,head=15)
    f.rect(231,657,167,153,'blue',None)
    f.robot(280,666,.91,'training-robot')
    for yy,t in [(745,'Shared'),(772,'Qwen3-4B'),(799,'backbone')]: f.text(314.5,yy,t,27,'b',BLUE,'middle')
    f.arrow([(400,728),(434,728)],width=4,head=13)
    f.rect(437,695,52,68,'blue','#59aeff',radius=6)
    f.subscript(463,741,'h','i',32,'i')
    f.arrow([(490,728),(519,728),(519,660),(549,660)],width=4,head=13)
    f.arrow([(490,728),(549,728)],width=4,head=13)
    f.arrow([(519,728),(519,787),(549,787)],width=4,head=13)
    for yy,t in [(639,'Correctness'),(701,'Error role'),(766,'Intrinsic type')]:
        f.rect(552,yy,179,43,'blue','#59aeff',radius=6)
        f.text(641.5,yy+29,t,25,color=BLUE,anchor='middle')
        f.arrow([(734,yy+21),(763,yy+21)],width=4,head=12)
    f.rect(766,639,137,43,'#ffffff','#bbbbbb',radius=5)
    f.text(834.5,668,'C / I',26,'i',anchor='middle')
    f.rect(766,701,137,43,'#ffffff','#bbbbbb',radius=5)
    f.text(834.5,719,'correct / new /',21,anchor='middle')
    f.text(834.5,739,'propagated',21,anchor='middle')
    f.rect(766,766,137,43,'#ffffff','#bbbbbb',radius=5)
    f.text(834.5,795,'8 types',26,anchor='middle')
    f.rect(954,658,146,65,'red','#ffb2b2',radius=6)
    f.text(1027,685,'Targets + masks',22,'b',anchor='middle',max_width=138)
    f.text(1027,711,'(from a)',22,'bi',BLUE,'middle')
    for yy in [660,722,787]: f.arrow([(954,690),(930,690),(930,yy),(906,yy)],width=3,head=12)
    f.text(1009,767,'Supervised',23,'i','#4d4d70','middle')
    f.text(1009,794,'targets',23,'i','#4d4d70','middle')
    f.rect(353,831,533,52,'blue',None)
    f.objective(619.5,867)
    f.text(619.5,909,'Type loss: known new errors only',25,'bi',BLUE,'middle')
    f.end()

    f.group('prm-utilization')
    f.rect(1125,591,536,335,stroke='#777777')
    f.text(1134,627,'(c) PRM Utilization',34,'b')
    f.rect(1134,637,518,125,dash=True)
    f.rect(1134,768,518,145,dash=True)
    f.text(1147,657,'Process Diagnosis',28,'bi',BLUE)
    f.stack(1161,670,81,85,name='diagnosis-pages')
    f.arrow([(1246,706),(1302,706)],width=4,head=14)
    f.robot(1308,670,.9,'diagnosis-robot')
    f.text(1342,744,'PRM',25,'b',BLUE,'middle')
    for yy,t in [(644,'Correctness'),(684,'Origin or propagation'),(724,'Intrinsic type')]:
        f.arrow([(1383,705),(1439,yy+17)],width=4,head=12)
        f.rect(1442,yy,196,33,'blue','#59aeff',radius=5)
        f.text(1540,yy+24,t,23,color=BLUE,anchor='middle',max_width=182)
    f.text(1147,792,'Weighted Answer Voting',28,'bi',BLUE)
    f.stack(1162,802,78,89,('s₁','⋯','sₙ',''),name='candidate-pages')
    f.text(1238,905,'Shared candidate pool',21,anchor='middle')
    f.arrow([(1243,834),(1272,834)],width=4,head=13)
    f.rect(1275,797,127,77,'blue','#59aeff',radius=5)
    f.text(1338.5,829,'Correctness',23,color=BLUE,anchor='middle',max_width=119)
    f.text(1338.5,855,'scores',23,color=BLUE,anchor='middle')
    f.arrow([(1403,834),(1416,834)],width=3,head=9)
    f.rect(1418,797,120,77,'blue','#59aeff',radius=5)
    for yy,t in [(822,'Group by'),(845,'answer +'),(867,'weighted vote')]: f.text(1478,yy,t,22,color=BLUE,anchor='middle',max_width=110)
    f.arrow([(1540,834),(1555,834)],width=3,head=9)
    f.rect(1558,797,83,77,'green','#88bd7b',radius=5)
    f.text(1599.5,830,'Selected',21,anchor='middle')
    f.text(1599.5,855,'answer',21,anchor='middle')
    f.end()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'assets')
    parser.add_argument('--png-width',type=int,default=5016)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fonts()
    fig=Figure(args.output_dir/'method_overview.pdf')
    build(fig)
    fig.save(args.output_dir/'method_overview.svg')
    import pypdfium2 as pdfium
    doc=pdfium.PdfDocument(str(args.output_dir/'method_overview.pdf'))
    page=doc[0]
    bitmap=page.render(scale=args.png_width/W)
    bitmap.to_pil().save(args.output_dir/'method_overview.png')
    bitmap.close(); page.close(); doc.close()
    print(f'Exported SVG, vector PDF, and {args.png_width}px-wide PNG to {args.output_dir}')


if __name__=='__main__': main()
