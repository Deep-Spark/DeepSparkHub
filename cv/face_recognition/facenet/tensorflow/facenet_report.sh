#!/bin/bash
# 该脚本针对Facenet单卡训练的结果日志处理，输出符合移动集采要求的模型信息。
# 模型源码的多卡训练，需要各厂商自行适配，该脚本需要适当修改

# 目标精度
target_acc=0.9905

echo "------------facenet report---------------"

# 一个epoch处理的总样本数
total_samples=`cat log/fps.txt | grep "Epoch: " | grep "Current Train Samples" | head -n 1 | awk '{print$6}'`
echo "samples(one epoch): ${total_samples}"
echo "target acc: ${target_acc}"

echo "--------first achieve target acc-----------"

# 首次达到目标精度时的精度结果
first_achieve_acc=`cat log/fps.txt | grep "Current Acc" | sort -n -k 6 | awk '$6 >= '${target_acc}'' | sort -n -k 2 | head -n 1 | awk '{print$6}'`
# 首次达到目标精度时的epoch数
current_epoch=`cat log/fps.txt | grep "Current Acc" | sort -n -k 6 | awk '$6 >= '${target_acc}'' | sort -n -k 2 | head -n 1 | awk '{print$2}'`
# 首次达到目标精度时的训练时长
current_train_time=`cat log/fps.txt | grep "Epoch: ${current_epoch}[[:blank:]]" | grep "All Train Time" | awk '{print$6}' | sort -n -k 1 -r | head -n 1`
# 首次达到目标精度时的评估时长
current_eval_time=`cat log/fps.txt | grep "Epoch: ${current_epoch}[[:blank:]]" | grep "All Eval Time" | awk '{print$6}' | head -n 1`
# 首次达到目标精度时的总时长
current_total_time=`cat log/fps.txt | grep "Epoch: ${current_epoch}[[:blank:]]" | grep "All Time" | awk '{print$5}' | head -n 1`
# 当前训练最大FPS
current_max_FPS=`cat log/fps.txt | awk '$2 <= '${current_epoch}'' | grep "Current Epoch FPS" | awk '{x[$2]+=$6}END{for(i in x){print i, x[i]}}' | sort -n -k 2 -r | head -n 1 | awk '{print$2}'`
# 当前训练平均FPS
current_average_FPS=`awk -v ts=${total_samples} -v eps=${current_epoch} -v ttt=${current_train_time} 'BEGIN{print(ts*(eps)/ttt)}'`
# 当前端到端平均FPS
current_e2e_FPS=`awk -v ts=${total_samples} -v eps=${current_epoch} -v ttt=${current_total_time} 'BEGIN{print(ts*(eps)/ttt)}'`

echo "first achieve target acc: ${first_achieve_acc}"
current_epoch=`awk -v ce=${current_epoch} 'BEGIN{print(ce+1)}'`
echo "current epoch: ${current_epoch}"
echo "current train time: ${current_train_time}"
echo "current eval time: ${current_eval_time}"
echo "current total time: ${current_total_time}"
echo "current max FPS: ${current_max_FPS}"
echo "current average FPS: ${current_average_FPS}"
echo "current e2e FPS: ${current_e2e_FPS}"


echo "------------achieve best acc---------------"
# 达到最优精度时的精度结果
best_acc=`cat log/fps.txt | grep "Current Acc" | sort -n -k 6 -r | awk '{print$6}' | head -n 1`
# 达到最优精度时的epoch数
best_acc_epoch=`cat log/fps.txt | grep "Current Acc" | sort -n -k 6 -r -k 2 | awk '{print$2}' | head -n 1`
# 达到最优精度时的训练时长
current_train_time=`cat log/fps.txt | grep "Epoch: ${best_acc_epoch}[[:blank:]]" | grep "All Train Time" | awk '{print$6}' | sort -n -k 1 -r | head -n 1`
# 首次最优精度时的评估时长
current_eval_time=`cat log/fps.txt | grep "Epoch: ${best_acc_epoch}[[:blank:]]" | grep "All Eval Time" | awk '{print$6}' | head -n 1`
# 首次最优精度时的总时长
current_total_time=`cat log/fps.txt | grep "Epoch: ${best_acc_epoch}[[:blank:]]" | grep "All Time" | awk '{print$5}' | head -n 1`
# 当前训练最大FPS
current_max_FPS=`cat log/fps.txt | awk '$2 <= '${best_acc_epoch}'' | grep "Current Epoch FPS" | awk '{x[$2]+=$6}END{for(i in x){print i, x[i]}}' | sort -n -k 2 -r | head -n 1 | awk '{print$2}'`
# 当前训练平均FPS
current_average_FPS=`awk -v ts=${total_samples} -v eps=${best_acc_epoch} -v ttt=${current_train_time} 'BEGIN{print(ts*(eps)/ttt)}'`
# 当前端到端平均FPS
current_e2e_FPS=`awk -v ts=${total_samples} -v eps=${best_acc_epoch} -v ttt=${current_total_time} 'BEGIN{print(ts*(eps)/ttt)}'`

echo "best acc: ${best_acc}"
best_acc_epoch=`awk -v ce=${best_acc_epoch} 'BEGIN{print(ce+1)}'`
echo "best acc epoch: ${best_acc_epoch}"
echo "current train time: ${current_train_time}"
echo "current eval time: ${current_eval_time}"
echo "current total time: ${current_total_time}"
echo "current max FPS: ${current_max_FPS}"
echo "current average FPS: ${current_average_FPS}"
echo "current e2e FPS: ${current_e2e_FPS}"


echo "-------------total time-------------------"
# 总epoch数
epoch_num=`cat log/fps.txt | sort -n -k 2 -r | head -n 1 | awk '{print$2}'`
# 最小的开始时间
start_time=`cat log/time.txt | grep "Start" | sort -n -k 3 | awk '{print$3}'`
# 最晚的结束时间
end_time=`cat log/time.txt | grep "End" | sort -n -k 3 -r | awk '{print$3}'`
# 程序运行总时长
all_time=`awk -v st=${start_time} -v et=${end_time} 'BEGIN{print(et-st)}'`

epoch_num=`awk -v ce=${epoch_num} 'BEGIN{print(ce+1)}'`
echo "total epoch number: ${epoch_num}"
echo "all time: ${all_time}"
