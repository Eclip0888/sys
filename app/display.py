"""Turn explanation dictionaries into simple rows the HTML templates can draw as bars."""
from .predictor import F, FEATURES, KEYS, fmt


def shap_view(e, inputs):
    c = e["contrib"]
    mx = max(abs(v) for v in c.values()) or 1
    rows = [{"label": F[k].label, "value": fmt(k, inputs[k]), "pp": c[k] * 100,
             "width": abs(c[k]) / mx * 100, "pos": c[k] > 0} for k in sorted(KEYS, key=lambda k: -abs(c[k]))]
    return {"base": e["base"] * 100, "prob": e["prob_pass"] * 100, "rows": rows}


def lime_view(e, inputs):
    cs = e["conditions"]
    mx = max(abs(x["weight"]) for x in cs) or 1
    rows = [{"rule": x["rule"], "value": fmt(x["feature"], inputs[x["feature"]]), "weight": x["weight"],
             "width": abs(x["weight"]) / mx * 100, "pos": x["weight"] > 0} for x in cs]
    return {"prob": e["prob_pass"] * 100, "rows": rows}


def cf_view(e, inputs):
    if not e.get("found"):
        return {"found": False}
    changed = {c["feature"] for c in e["changes"]}
    rows = [{"label": f.label, "now": fmt(f.key, inputs[f.key]), "cf": fmt(f.key, e["new_inputs"][f.key]),
             "changed": f.key in changed, "fixed": not f.actionable} for f in FEATURES]
    sentence = " and ".join(f"{F[c['feature']].label.lower()} were {fmt(c['feature'], c['new'])} "
                            f"instead of {fmt(c['feature'], c['old'])}" for c in e["changes"])
    return {"found": True, "rows": rows, "sentence": sentence, "label": e["label"], "new_label": e["new_label"],
            "new_prob": e["new_prob"] * 100}


def view(kind, e, inputs):
    return {"SHAP": shap_view, "LIME": lime_view, "CF": cf_view}[kind](e, inputs)


def inputs_text(inputs):
    return ", ".join(f"{F[k].label.lower()} {fmt(k, inputs[k])}" for k in KEYS)
