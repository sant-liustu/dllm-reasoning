#!/bin/bash

# 完整测试套件运行脚本
# 运行所有 interleaved training 相关测试

LOG_DIR="dllm_reasoning/logs"
DATE=$(date +"%Y%m%d_%H%M%S")

echo "======================================================================================================"
echo "运行完整测试套件"
echo "日志目录: $LOG_DIR"
echo "======================================================================================================"

# 测试 1: 标签对齐验证
echo ""
echo "===== 测试 1/5: 标签对齐验证 ====="
python dllm_reasoning/test_list/trainer/test_label_alignment.py 2>&1 | tee $LOG_DIR/${DATE}_test_label_alignment.log
TEST1_STATUS=$?

# 测试 2: Mask 隔离性验证（float64精度）
echo ""
echo "===== 测试 2/5: Mask 隔离性验证 ====="
python dllm_reasoning/test_list/trainer/test_mask_isolation_fp64.py 2>&1 | tee $LOG_DIR/${DATE}_test_mask_isolation.log
TEST2_STATUS=$?

# 测试 3: 梯度反向传播
echo ""
echo "===== 测试 3/5: 梯度反向传播 ====="
python dllm_reasoning/test_list/trainer/test_gradient_backward.py 2>&1 | tee $LOG_DIR/${DATE}_test_gradient_backward.log
TEST3_STATUS=$?

# 测试 4: Batch Padding
echo ""
echo "===== 测试 4/5: Batch Padding ====="
python dllm_reasoning/test_list/trainer/test_batch_padding.py 2>&1 | tee $LOG_DIR/${DATE}_test_batch_padding.log
TEST4_STATUS=$?

# 测试 5: 小规模过拟合
echo ""
echo "===== 测试 5/5: 小规模过拟合 ====="
python dllm_reasoning/test_list/trainer/test_overfit_single_sample.py 2>&1 | tee $LOG_DIR/${DATE}_test_overfit_single_sample.log
TEST5_STATUS=$?

# 总结
echo ""
echo "======================================================================================================"
echo "测试套件完成"
echo "======================================================================================================"
echo ""

echo "测试结果："
echo "  1. 标签对齐验证:      $([ $TEST1_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo "  2. Mask 隔离性验证:   $([ $TEST2_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo "  3. 梯度反向传播:      $([ $TEST3_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo "  4. Batch Padding:     $([ $TEST4_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo "  5. 小规模过拟合:      $([ $TEST5_STATUS -eq 0 ] && echo '✅ 通过' || echo '❌ 失败')"
echo ""

# 检查是否所有测试都通过
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ] && [ $TEST3_STATUS -eq 0 ] && [ $TEST4_STATUS -eq 0 ] && [ $TEST5_STATUS -eq 0 ]; then
    echo "✅ 所有测试通过！可以开始大规模训练。"
    echo ""
    exit 0
else
    echo "❌ 部分测试失败！请检查日志并修复问题。"
    echo ""
    exit 1
fi
