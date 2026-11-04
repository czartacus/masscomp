"""Print shell `export` lines for the sampler knobs in a spec string ("regrow_p=0.1,init=lattice"), using
the same names as studies/diag.py; the study workflow evals this so one `knobs` input covers every
MASSCOMP_* switch. Usage: eval "$(python studies/knobs.py "$KNOBS")" """
import sys
from studies.diag import KNOBS, parse

if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else ""
    for k, v in parse(spec).items():
        if k in KNOBS:
            print(f"export {KNOBS[k]}='{v.replace('+', ',')}'")
        elif k.startswith("env_"):
            print(f"export MASSCOMP_{k[4:].upper()}='{v.replace('+', ',')}'")
