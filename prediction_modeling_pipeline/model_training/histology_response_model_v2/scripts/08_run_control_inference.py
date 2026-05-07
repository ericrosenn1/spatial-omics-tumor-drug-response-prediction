"""
Script: 08_run_control_inference.py

Purpose:
    Run blank and noise image controls through the trained image_treatment model.

Pipeline role:
    Step 08 of histology_response_model_v2. This post-training sanity check
    loads the trained image_treatment model, creates non-informative blank and
    random-noise images, evaluates them across treatment embeddings, and writes
    control prediction summaries.

Scientific context:
    Blank/noise controls test whether model outputs remain stable on images that
    should not contain histologic response signal. These controls are audit
    outputs, not training data, and they inform how conservatively downstream
    teacher_builder should use histology probabilities.

Documentation safety:
    Documentation edits should not change executable behavior, thresholds, paths,
    schemas, model settings, or outputs.
"""


# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

from histology_model_v2_lib import load_yaml, output_root, ensure_dir, read_table

# =============================================================================
# Dynamic import of Step 07 model class
# =============================================================================

import importlib.util
_spec = importlib.util.spec_from_file_location("trainmod", Path(__file__).parent / "07_train_baselines_and_conditioned_model.py")
trainmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainmod)
HistModel = trainmod.HistModel



# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap=argparse.ArgumentParser(); ap.add_argument("--config", required=True); args=ap.parse_args()
    cfg=load_yaml(args.config); out=ensure_dir(output_root(cfg) / "08_controls")
    # Controls are run only after the selected image_treatment checkpoint exists.
    model_root=output_root(cfg) / "07_models" / "image_treatment"
    ckpt_path=model_root / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(ckpt_path, map_location=device)
    # The model is reconstructed with the exact training configuration stored in the checkpoint.
    train_cfg=ckpt["training_cfg"]
    model=HistModel("image_treatment", ckpt["n_treatments"], ckpt["n_classes"], train_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    # Treatment encoders are reused so controls are evaluated across the trained treatment embedding space.
    enc=json.loads((output_root(cfg)/"07_models"/"encoders.json").read_text())
    inv_treat={int(v):k for k,v in enc["treatment_encoder"].items()}
    size=int(train_cfg["image_size"])
    tf=transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    # Blank and noise images are deliberately non-informative controls.
    blank=Image.new("RGB", (size,size), (255,255,255))
    rng=np.random.default_rng(42)
    noise=Image.fromarray(rng.integers(0,255,size=(size,size,3), dtype=np.uint8))
    rows=[]
    # Each control image is paired with every treatment embedding to reveal treatment-driven output spread.
    for img_name,img in [("blank", blank), ("noise", noise)]:
        x=tf(img).unsqueeze(0).to(device)
        for tidx,tkey in inv_treat.items():
            tid=torch.tensor([tidx], dtype=torch.long).to(device)
            with torch.no_grad():
                prob=torch.softmax(model(x, tid), dim=1).detach().cpu().numpy()[0]
            rows.append({"control": img_name, "treatment_id": tidx, "treatment_key": tkey, "prob_RESPONDER": float(prob[1] if len(prob)>1 else prob[0])})
    controls=pd.DataFrame(rows)
    controls.to_csv(out / "blank_noise_control_predictions.tsv", sep="\t", index=False)
    # The compact summary is the downstream audit view of non-informative-image behavior.
    summary=controls.groupby("control")["prob_RESPONDER"].agg(["mean","std","min","max"]).reset_index()
    summary.to_csv(out / "control_summary.tsv", sep="\t", index=False)
    print("DONE")
    print(out)

if __name__ == "__main__":
    main()
