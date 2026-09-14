"""Read Circuit Training netlist hyperedges and pin offsets without TensorFlow."""
from pathlib import Path
import re
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, text_format


def graph_type():
    f=descriptor_pb2.FileDescriptorProto(name='science_netlist.proto',package='science',syntax='proto2')
    a=f.message_type.add(name='Attr')
    for number,(name,kind) in enumerate([('placeholder',9),('f',2),('i',3),('b',8),('s',12)],1):
        a.field.add(name=name,number=number,type=kind,label=1)
    n=f.message_type.add(name='Node')
    n.field.add(name='name',number=1,type=9,label=1)
    n.field.add(name='input',number=2,type=9,label=3)
    entry=n.nested_type.add(name='Attributes');entry.options.map_entry=True
    entry.field.add(name='key',number=1,type=9,label=1)
    entry.field.add(name='value',number=2,type=11,type_name='.science.Attr',label=1)
    n.field.add(name='attr',number=3,type=11,type_name='.science.Node.Attributes',label=3)
    g=f.message_type.add(name='Graph');g.field.add(name='node',number=1,type=11,type_name='.science.Node',label=3)
    pool=descriptor_pool.DescriptorPool();pool.Add(f)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('science.Graph'))


def parse(netlist,initial, *, require_legal_grid=True):
    graph=graph_type()();text_format.Parse(Path(netlist).read_text(),graph)
    nodes=[]
    for i,n in enumerate(graph.node):
        attrs={k:next((getattr(v,field.name) for field,_ in v.ListFields()),None) for k,v in n.attr.items()}
        nodes.append(dict(id=i,name=n.name,inputs=list(n.input),**attrs))
    if len({n['name'] for n in nodes})!=len(nodes):raise ValueError('duplicate node names')
    text=Path(initial).read_text()
    wh=re.search(r'Width\s*:\s*([\d.]+)\s+Height\s*:\s*([\d.]+)',text)
    cr=re.search(r'Columns\s*:\s*(\d+)\s+Rows\s*:\s*(\d+)',text)
    if not wh or not cr:raise ValueError('initial placement must specify canvas and grid')
    width,height=map(float,wh.groups());cols,rows=map(int,cr.groups())
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):continue
        i,x,y,o,fixed=line.split()
        node=nodes[int(i)];node.update(x=float(x),y=float(y),orientation=o,fixed=bool(int(fixed)))
    blockages=[list(map(float,m.groups())) for m in re.finditer(r'# Blockage\s*:\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)',text)]
    by_name={n['name']:n for n in nodes}
    movable=[];fixed=[];clusters=[];pins=[];nets=[]
    for node in nodes:
        kind=node.get('type','').upper()
        soft=node.get('type')=='macro'
        if kind=='MACRO_PIN':
            parent=node.get('macro_name')
            if parent not in by_name:raise ValueError('pin parent not found')
            pins.append(dict(id=node['id'],macro_id=by_name[parent]['id'],
                             x_offset=node.get('x_offset',0),y_offset=node.get('y_offset',0)))
        elif kind in ('MACRO','STDCELL'):
            if node.get('fixed'):
                w,h=node['width'],node['height'];x,y=node['x'],node['y']
                if node.get('orientation') in ('E','W','FE','FW'):w,h=h,w
                fixed.append(dict(id=node['id'],bounds=[x-w/2,y-h/2,x+w/2,y+h/2],orientation=node.get('orientation','N')))
            elif soft or kind=='STDCELL':clusters.append(node)
            else:
                orient=node.get('orientation','N')
                movable.append(dict(id=node['id'],width=node['width'],height=node['height'],
                    allowed_orientations=['N','FN','S','FS'] if orient in ('N','FN','S','FS') else ['E','FE','W','FW'],
                    x=node['x'],y=node['y'],orientation=orient))
        elif kind=='PORT':fixed.append(dict(id=node['id'],x=node['x'],y=node['y']))
        if node['inputs']:
            nets.append(dict(pin_ids=[node['id']]+[by_name[name]['id'] for name in node['inputs']],weight=node.get('weight',1.)))
    cells=[]
    for m in movable:
        c=int(m['x']/width*cols);r=int(m['y']/height*rows)
        if require_legal_grid and (abs((c+.5)*width/cols-m['x'])>1e-5 or abs((r+.5)*height/rows-m['y'])>1e-5):
            raise ValueError('baseline macros must be centered on grid cells')
        cells.append(r*cols+c)
    problem=dict(width=width,height=height,num_cols=cols,num_rows=rows,movable_macro_ids=[m['id'] for m in movable],
        movable_macros=movable,fixed_objects=fixed,standard_cell_clusters=clusters,pins=pins,nets=nets,
        nodes=nodes,blockages=blockages,routing_capacity=dict(horizontal=70.33,vertical=74.51,macro_horizontal=51.79,macro_vertical=51.79),
        initial_placement=dict(cell_ids=cells,orientations=[m['orientation'] for m in movable]))
    from .contracts import placement
    if require_legal_grid:placement(problem,problem['initial_placement'])
    return problem
