"""Use the established identity/checksum/receipt guarded submitter for these retries."""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('campaign_submit', HERE.parent/'fresh-grpo-campaign-20260919/submit.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HERE=HERE
if __name__=='__main__':
    module.main()
