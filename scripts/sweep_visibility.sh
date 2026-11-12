#!/usr/bin/env bash
# Visibility/protection frontier on the 20 validation images.
cd /workspace/DiffVax && source /workspace/venv/bin/activate && source /root/.discord_env && export HF_HOME=/workspace/hf-cache
run () {  # tag eps extra...
  tag=$1; eps=$2; shift 2
  python scripts/make_teacher.py --splits validation --epsilon $eps --lpips-weight 0 "$@" --out /workspace/outputs/sweep_$tag
  python scripts/eval_protocol.py --protected-dir /workspace/outputs/sweep_$tag/validation/images --control-eps $eps \
     --editors "FLUX.2 Klein" --resolution 512 --editor-resolution 512 --max-images 20 --seeds 0 --identity \
     --output-dir /workspace/outputs/protocol_sweep_$tag
  python - "$tag" <<'PY'
import json,os,sys,requests
from PIL import Image
tag=sys.argv[1]; d=f"/workspace/outputs/protocol_sweep_{tag}"
md=open(d+"/protocol_report.md").read()
# 2x zoom crop of one protected image next to its original
name="brooke-cagle-pCdYJUAMSCo-unsplash"
o=Image.open(f"data/validation/images/{name}.png").convert("RGB").resize((512,512),Image.LANCZOS)
p=Image.open(f"/workspace/outputs/sweep_{tag}/validation/images/{name}.png").convert("RGB")
z=Image.new("RGB",(1024,512)); z.paste(o.crop((128,64,384,320)).resize((512,512),Image.NEAREST),(0,0)); z.paste(p.crop((128,64,384,320)).resize((512,512),Image.NEAREST),(512,0)); z.save(d+"/zoom.png")
url=os.environ["DISCORD_WEBHOOK"]
for f,cap in ((d+"/samples_FLUX_2_Klein.png",f"**sweep {tag}**\n"+md),(d+"/zoom.png",f"**sweep {tag}** 2x zoom: original | protected")):
    with open(f,"rb") as fh:
        requests.post(url,data={"payload_json":json.dumps({"content":cap[:1900]})},files={"file":(os.path.basename(f),fh,"image/png")},timeout=60)
print("SWEEP_DONE", tag, flush=True)
PY
}
run eps8_ssim95 8 --ssim-target 0.95
run eps8_ssim97 8 --ssim-target 0.97
run eps4 4
echo ALL_SWEEPS_DONE
