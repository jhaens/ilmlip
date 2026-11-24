# Fine-tune the bundled foundation model on 10 AIMD frames of Li13Si4 (204 atoms) in each of
# the four ways. test.xyz, 20 frames of another trajectory, is evaluated before and after.
set -e
cd "$(dirname "$0")"

# all weights
python -m ilmlip.finetune --foundation OAM-S-omat --train train.xyz --test test.xyz --out naive
# low-rank updates of frozen weights
python -m ilmlip.finetune --foundation OAM-S-omat --train train.xyz --test test.xyz --out lora --mode lora
# the same architecture, random weights
python -m ilmlip.finetune --foundation OAM-S-omat --train train.xyz --test test.xyz --out scratch --mode scratch
# one head for train.xyz, one for replay structures from MPtrj (downloads 600 MB once)
python -m ilmlip.finetune --foundation OAM-S-omat --train train.xyz --test test.xyz --out multihead --mode multihead
