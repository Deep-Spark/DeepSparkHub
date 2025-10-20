
COCO_PATH="`pwd`/../../data/datasets/coco2017"

: ${BATCH_SIZE:=160}

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

cd ../../../cv/detection/ssd/pytorch/

echo "python3 train.py --no-dali --dali-cache 0 --data=${COCO_PATH} \
--batch-size=${BATCH_SIZE} --warmup-factor=0 --warmup=650 --lr=2.92e-3 --threshold=0.08 --epochs 5 --eval-batch-size=160 \
--wd=1.6e-4 --use-fp16 --delay-allreduce --lr-decay-factor=0.2 --lr-decay-epochs 34 45 --opt-level O2 --seed 1769250163"

python3 train.py --dali --dali-cache 0 --data=${COCO_PATH} \
--batch-size=${BATCH_SIZE} --warmup-factor=0 --warmup=650 --lr=2.92e-3 --threshold=0.08 --epochs 5 --eval-batch-size=160 \
--wd=1.6e-4 --use-fp16 --jit --nhwc --pad-input --delay-allreduce --lr-decay-factor=0.2 --lr-decay-epochs 34 45 --opt-level O2 --seed 1769250163 "$@";  check_status

cd -
exit ${EXIT_STATUS}
