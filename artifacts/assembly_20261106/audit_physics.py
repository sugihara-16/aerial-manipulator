"""Use a geometry-free coordinate shell; audit the saved full physical XML."""
import json,sys
from scripts import report_assembly_panel as audit
factory=audit.make_assembly_plant
def shell(*args,**kwargs):
    kwargs['meshes']=False
    return factory(*args,**kwargs)
audit.make_assembly_plant=shell
r=audit.report(sys.argv[1])
print(json.dumps({k:v for k,v in r.items()if k not in ('checks','contrasts')},indent=2))
