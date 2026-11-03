"""Prepare one EML-family operator: minimal-construction table, scout targets, calculator targets."""
import argparse
import json

from reach.family import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
a = ap.parse_args()
mt = prepare(a.grammar)
found = sum(h is not None for h in mt["table"].values())
print("RESULT " + json.dumps(dict(grammar=a.grammar, terminal_state="completed", ceiling=mt["ceiling"],
                                  calc_found=found)), flush=True)
